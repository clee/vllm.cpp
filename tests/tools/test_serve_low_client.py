"""Pinned-client wrapper tests derived from SGLang bench functionality tests.

Sources: ``test_bench_serving_functionality.py`` and
``bench_serving.py:232-344,1588-1675`` @ SGLang 28b095c.
"""

from __future__ import annotations

import contextlib
import http.server
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

from tools.bench import run_serve_low
from tools.bench.run_serve_low import (
    MODEL_KEYS,
    BenchRun,
    build_bench_command,
    build_dry_run_manifest,
    openai_stream_probe,
    openai_usage_preflight,
    run_usage_batch,
    validate_raw_result,
)
from tools.bench.serve_low_common import HarnessError, SGLANG_IMAGE


class _CompletionHandler(http.server.BaseHTTPRequestHandler):
    lock = threading.Lock()
    active = 0
    peak = 0
    payloads: list[dict] = []

    def log_message(self, *_args) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        with self.lock:
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            type(self).payloads.append(payload)
        try:
            if self.path == "/error":
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"fixture"}')
                return
            if self.path == "/diagnostic-error":
                body = json.dumps(
                    {
                        "error": {
                            "message": "vt: sentinel root cause",
                            "type": "InternalServerError",
                            "code": 500,
                        }
                    }
                ).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                for token in "ABCD":
                    body = json.dumps({"choices": [{"text": token}]})
                    self.wfile.write(f"data: {body}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.015)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
                return
            time.sleep(0.02)
            body = json.dumps(
                {
                    "choices": [{"finish_reason": "length", "text": "ABCD"}],
                    "usage": {
                        "completion_tokens": 4,
                        "prompt_tokens": 8,
                        "total_tokens": 12,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            with self.lock:
                type(self).active -= 1


class ClientTests(unittest.TestCase):
    def setUp(self) -> None:
        _CompletionHandler.active = 0
        _CompletionHandler.peak = 0
        _CompletionHandler.payloads = []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_request_shape_concurrency_cap_and_error_propagation(self) -> None:
        results = run_usage_batch(
            self.base + "/v1/completions",
            [f"prompt {index}" for index in range(6)],
            max_concurrency=2,
            prompt_tokens=8,
            completion_tokens=4,
        )
        self.assertEqual(len(results), 6)
        self.assertGreaterEqual(_CompletionHandler.peak, 2)
        self.assertLessEqual(_CompletionHandler.peak, 2)
        payload = _CompletionHandler.payloads[0]
        self.assertEqual(payload["max_tokens"], 4)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertTrue(payload["ignore_eos"])
        self.assertFalse(payload["stream"])
        with self.assertRaises(HarnessError):
            openai_usage_preflight(
                self.base + "/error",
                "prompt",
                prompt_tokens=8,
                completion_tokens=4,
            )

    def test_stream_probe_requires_incremental_exact_chunk_count(self) -> None:
        result = openai_stream_probe(
            self.base + "/v1/completions",
            "prompt",
            completion_tokens=4,
            minimum_spread_s=0.02,
        )
        self.assertEqual(result.emitted_chunks, 4)
        self.assertEqual(result.generated_text, "ABCD")
        self.assertGreater(result.spread_s, 0.02)

    def test_raw_detail_validation_is_fail_closed(self) -> None:
        record = {
            "completed": 2,
            "errors": ["", ""],
            "generated_texts": ["ABCD", "ABCD"],
            "input_lens": [8, 8],
            "itls": [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
            "output_lens": [4, 4],
            "ttfts": [0.2, 0.2],
        }
        validate_raw_result(record, expected_requests=2, prompt_len=8, output_len=4)
        record["errors"][1] = "boom"
        with self.assertRaises(HarnessError):
            validate_raw_result(record, expected_requests=2, prompt_len=8, output_len=4)

    def test_pinned_client_command_and_dry_run_refuse_floating_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "evidence"
            (root / "models").mkdir()
            corpus = evidence / "corpus" / "27"
            corpus.mkdir(parents=True)
            (corpus / "c1-r1.jsonl").write_text("{}\n")
            run = BenchRun(
                image=SGLANG_IMAGE,
                model_repo=root / "models",
                model_revision="revision",
                evidence_root=evidence,
                model_key="27",
                engine="ours",
                base_url="http://127.0.0.1:30000",
                concurrency=1,
                repetition=1,
            )
            command = build_bench_command(run)
            self.assertIn("--pull=never", command)
            self.assertNotIn("--gpus", command)
            self.assertEqual(command.count("sglang.bench_serving"), 1)
            manifest = build_dry_run_manifest(
                claim_root=root,
                vllm_cpp_sha="a" * 40,
                image=SGLANG_IMAGE,
            )
            self.assertTrue(manifest["dry_run"])
            self.assertIn("native_output_id_parity", manifest["pending_preconditions"])
            with self.assertRaises(HarnessError):
                build_dry_run_manifest(
                    claim_root=root,
                    vllm_cpp_sha="a" * 40,
                    image="sglang:latest",
                )

    def test_diagnostic_error_body_captures_500_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                json.dumps({"conversations": [{"content": "prompt zero"}]}) + "\n"
            )
            output = root / "r1-error-body.json"
            record = run_serve_low.capture_diagnostic_error_body(
                self.base + "/diagnostic-error",
                corpus,
                output,
            )
            self.assertEqual(record["status"], 500)
            self.assertTrue(record["diagnostic"])
            self.assertEqual(record["mode"], "diagnostic-c16")
            self.assertEqual(
                record["body"]["error"]["message"], "vt: sentinel root cause"
            )

            written = json.loads(output.read_text())
            self.assertEqual(written["status"], 500)
            self.assertEqual(
                written["body"]["error"]["message"], "vt: sentinel root cause"
            )
            self.assertTrue(written["diagnostic"])
            self.assertEqual(written["mode"], "diagnostic-c16")

            # Fail-closed: refuse to overwrite an existing capture.
            with self.assertRaises(HarnessError):
                run_serve_low.capture_diagnostic_error_body(
                    self.base + "/diagnostic-error",
                    corpus,
                    output,
                )

            # The non-streaming completion payload is what the server saw.
            sent = _CompletionHandler.payloads[-1]
            self.assertFalse(sent["stream"])
            self.assertEqual(sent["max_tokens"], 128)
            self.assertTrue(sent["ignore_eos"])
            self.assertEqual(sent["temperature"], 0.0)

    def test_campaign_shell_dry_run_creates_manifest_without_gpu_work(self) -> None:
        repo = pathlib.Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            claim = pathlib.Path(temporary) / "claim"
            subprocess.run(
                [
                    str(repo / "scripts" / "dgx-sglang-low-concurrency.sh"),
                    "--dry-run",
                    "--claim-root",
                    str(claim),
                    "--vllm-cpp-sha",
                    "b" * 40,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(
                (claim / "evidence" / ("b" * 40) / "manifest.json").read_text()
            )
            self.assertTrue(manifest["dry_run"])
            self.assertIn("host_idle_proof", manifest["pending_preconditions"])
            self.assertEqual(manifest["gpu_lock_acquisitions_planned"], 1)
            self.assertFalse(manifest["pull_under_gpu_lock"])
            self.assertIn("--pull=never", manifest["planned_commands"]["client"])


def _synthetic_raw_result(
    *,
    requests: int = 80,
    prompt_len: int = 1024,
    output_len: int = 128,
    max_concurrency: int = 1,
) -> dict:
    """The smallest record `validate_raw_result` accepts as a complete leg."""

    return {
        "completed": requests,
        "errors": [""] * requests,
        "generated_texts": ["x"] * requests,
        "input_lens": [prompt_len] * requests,
        "itls": [[0.001] * (output_len - 1) for _ in range(requests)],
        "max_concurrent_requests": max_concurrency,
        "num_prompts": requests,
        "output_lens": [output_len] * requests,
        "ttfts": [0.01] * requests,
    }


def _container_to_host(container: str, mounts: dict[str, pathlib.Path]) -> pathlib.Path:
    """Resolve an in-container path through the bind mounts of the same command.

    `BenchRun.corpus_path` / `BenchRun.output_path` and the `container_corpus`
    / `container_output` strings in `build_bench_command` derive from
    `model_key` INDEPENDENTLY.  Translating one back through the mount the same
    command declared is what makes a disagreement between the two derivations
    observable instead of silent.
    """

    for destination, source in sorted(mounts.items(), key=lambda item: -len(item[0])):
        if container == destination or container.startswith(destination + "/"):
            return source / container[len(destination):].lstrip("/")
    raise AssertionError(f"{container} is not under any declared bind mount")


class _FakeCompletedProcess:
    returncode = 0


class ModelKeyEvidenceRoutingTest(unittest.TestCase):
    """A named subject must own its evidence, and only its own (#1594).

    `--model-key` is a label that keys the evidence tree, so the failure this
    pins is not a crash.  It is a run that completes, writes a plausible raw
    result, and files it under ANOTHER subject's key -- which is why asserting
    the parser's `choices` would not be enough.  The assertions below enter
    through `main()` on a real argv and read the paths the wrapper actually
    derived, so a key accepted at the parser but not threaded to the corpus and
    raw paths fails here.
    """

    def _run_bench_through_main(
        self, root: pathlib.Path, model_key: str, engine: str = "ours"
    ) -> list[str]:
        evidence = root / "evidence"
        models = root / "models"
        models.mkdir(exist_ok=True)
        corpus = evidence / "corpus" / model_key
        corpus.mkdir(parents=True, exist_ok=True)
        (corpus / "c1-r1.jsonl").write_text("{}\n")
        captured: list[list[str]] = []

        def fake_run(command, check=False):  # mirrors subprocess.run's call
            captured.append(list(command))
            mounts: dict[str, pathlib.Path] = {}
            for index, item in enumerate(command):
                if item != "--mount":
                    continue
                fields = dict(
                    field.split("=", 1)
                    for field in command[index + 1].split(",")
                    if "=" in field
                )
                mounts[fields["dst"]] = pathlib.Path(fields["src"])
            dataset = command[command.index("--dataset-path") + 1]
            self.assertTrue(_container_to_host(dataset, mounts).is_file())
            written = _container_to_host(
                command[command.index("--output-file") + 1], mounts
            )
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_text(json.dumps(_synthetic_raw_result()) + "\n")
            return _FakeCompletedProcess()

        argv = [
            "run_serve_low.py",
            "bench",
            "--model-repo", str(models),
            "--model-revision", "36f717a22990e82c54c1d48ee77c491b87825680",
            "--evidence", str(evidence),
            "--model-key", model_key,
            "--engine", engine,
            "--base-url", "http://127.0.0.1:30000",
            "--concurrency", "1",
            "--repetition", "1",
        ]
        with unittest.mock.patch.object(run_serve_low.subprocess, "run", fake_run):
            with unittest.mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(run_serve_low.main(), 0)
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_each_admitted_key_owns_its_corpus_and_raw_tree(self) -> None:
        self.assertIn("q38mtp", MODEL_KEYS)
        for model_key in MODEL_KEYS:
            with self.subTest(model_key=model_key):
                with tempfile.TemporaryDirectory() as temporary:
                    root = pathlib.Path(temporary)
                    command = self._run_bench_through_main(root, model_key)
                    evidence = root / "evidence"
                    dataset = command[command.index("--dataset-path") + 1]
                    output = command[command.index("--output-file") + 1]

                    # The in-container paths `build_bench_command` hands
                    # the pinned client.
                    self.assertEqual(dataset, f"/evidence/corpus/{model_key}/c1-r1.jsonl")
                    self.assertEqual(
                        output, f"/evidence/raw/{model_key}/ours/c1-r1.jsonl"
                    )

                    # The host paths `BenchRun` derives.  The raw file
                    # exists only because the container path resolved back to
                    # it through the command's own bind mount.
                    self.assertTrue(
                        (
                            evidence / "raw" / model_key / "ours" / "c1-r1.jsonl"
                        ).is_file()
                    )

                    # No OTHER admitted subject's tree was touched or named.
                    for other in MODEL_KEYS:
                        if other == model_key:
                            continue
                        self.assertNotIn(f"/{other}/", dataset)
                        self.assertNotIn(f"/{other}/", output)
                        self.assertFalse((evidence / "raw" / other).exists())
                        self.assertFalse((evidence / "corpus" / other).exists())

    def test_the_third_key_is_not_a_spelling_of_an_existing_one(self) -> None:
        """`27` already names a DIFFERENT 27B checkpoint, so the keys must not
        collide as substrings of each other's evidence paths."""

        for model_key in MODEL_KEYS:
            for other in MODEL_KEYS:
                if other == model_key:
                    continue
                with self.subTest(model_key=model_key, other=other):
                    self.assertNotIn(other, model_key)

    def test_an_unadmitted_key_is_refused_at_both_the_parser_and_the_command(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            evidence = root / "evidence"
            (root / "models").mkdir()
            corpus = evidence / "corpus" / "38"
            corpus.mkdir(parents=True)
            (corpus / "c1-r1.jsonl").write_text("{}\n")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as parser_refusal:
                    run_serve_low._parser().parse_args(
                        [
                            "bench",
                            "--model-repo", str(root / "models"),
                            "--model-revision", "revision",
                            "--evidence", str(evidence),
                            "--model-key", "38",
                            "--engine", "ours",
                            "--base-url", "http://127.0.0.1:30000",
                            "--concurrency", "1",
                            "--repetition", "1",
                        ]
                    )
            self.assertEqual(parser_refusal.exception.code, 2)

            # The library entry refuses the same key, so a caller that bypasses
            # the parser cannot open an evidence tree nobody declared.
            run = BenchRun(
                image=SGLANG_IMAGE,
                model_repo=root / "models",
                model_revision="revision",
                evidence_root=evidence,
                model_key="38",
                engine="ours",
                base_url="http://127.0.0.1:30000",
                concurrency=1,
                repetition=1,
            )
            with self.assertRaises(HarnessError):
                build_bench_command(run)


if __name__ == "__main__":
    unittest.main()
