// THE RENDER PHASE INSTRUMENT, held to what its own numbers claim.
//
// Row LTX25-PHASE-INSTRUMENT, issues #1668, #1569, #1571. Spec:
// `.agents/specs/ltx25-phase-instrument.md`.
//
// WHY THIS FILE IS NOT IN `test_ltx2_video.cpp`. The cases below are about the
// INSTRUMENT rather than about LTX-2.5: they build synthetic timelines whose
// leaves contain a `sleep` and nothing else, because the question is where a
// charge LANDS and what the emitter writes, not what a render does. Two of them
// need a table of thousands of records, which no render produces. Keeping them
// beside a 5000-line model suite that renders a fixture would make an instrument
// question cost a model build, and it would put them in the file three other
// issues are actively editing.
//
// WHAT IS STILL GATED IN `test_ltx2_video.cpp`, and has to be. Everything here
// calls `PhaseLog` directly, which proves the class works and never that a
// render reaches it. The reachability half — that `vllm_video_generate` emits a
// table carrying `instrument_seconds` and `gaps` — is asserted on the table the
// ABI writes, in `a render through the ABI emits a phase table that SUMS to
// wall`. Neither file is sufficient alone.
//
// THE ONE NUMBER THIS FILE REFUSES TO ASSERT is a residue measured against the
// instrument's own charge. `.agents/specs/ltx25-phase-residue.md` `## Design` 3
// records three fresh reviews measuring `residue <= 2 * instrument` red 4 times
// in 45 runs at load 88 (max 4.115) and 28 times in 160 at load 125 (max 5.55),
// because the UN-instrumented remainder of a boundary dilates faster than the
// instrumented part when the box slows. Every bound below is either an
// accounting identity, which no scheduler can move, or a one-sided comparison
// whose noise can only push it AWAY from red. Read that section before adding a
// ratio here.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include "vllm/multimodal/render_phase_log.h"

namespace {

namespace phase = vllm::multimodal::phase;

std::string ReadAll(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

// A temporary directory and the table written inside it, removed on the way out.
// Every case here writes a real file because `WriteJson` is what is under test:
// the emitted table is the artifact a reader gets, and asserting on an in-memory
// record vector would skip the half of the code these cases exist for.
class TableFile {
 public:
  TableFile() {
    std::snprintf(dir_, sizeof(dir_), "/tmp/vllm_phase_unit_XXXXXX");
    REQUIRE(::mkdtemp(dir_) != nullptr);
    path_ = std::string(dir_) + "/phase-log.json";
  }
  ~TableFile() {
    ::unlink(path_.c_str());
    ::rmdir(dir_);
  }
  TableFile(const TableFile&) = delete;
  TableFile& operator=(const TableFile&) = delete;

  const std::string& path() const { return path_; }

  nlohmann::json Write(const phase::PhaseLog& log) const {
    std::string why;
    REQUIRE_MESSAGE(log.WriteJson(path_, "unit", "cpu", &why), why);
    return nlohmann::json::parse(ReadAll(path_));
  }

 private:
  char dir_[64] = {};
  std::string path_;
};

void SleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

}  // namespace

// ─── the instrument charges its OWN cost to the right place (#1668) ──────────
//
// The rule under test is one sentence from `render_phase_log.cpp`: every
// interval of the instrument's own wall is charged to the innermost live
// NON-SPAN record at the moment it is spent, and to the table when none is live.
// Three consequences, and each one is a different defect if it is wrong:
//
//   * A CHILD'S BOUNDARY IS THE PARENT'S COST. Opening and closing a nested
//     scope costs wall that lies inside the parent and outside the child, which
//     is precisely the uncovered time the coverage gate reads. Charged to the
//     table instead, that gate would have nothing to subtract and the number
//     would say the parent encloses a phase nobody named.
//   * A BOUNDARY WITH NOTHING LIVE IS THE TABLE'S COST. That is the residue the
//     sum gate reads, and it is the whole of `unaccounted_seconds`'s
//     explanation.
//   * A SPAN IS NOT A LEAF. `Sum` skips spans, so time inside a span and outside
//     every leaf IS the residue; charging it to the enclosing span would hide it
//     in a number nothing adds up. This is the case the LTX-2.5 driver actually
//     hits, because `load` and `generate` are spans that stay open across
//     everything beneath them.
//
// NOTHING HERE IS A DURATION COMPARISON. The three assertions are "it moved",
// "it did not move at all" and "it is positive", which is why a loaded box
// cannot change the verdict.
TEST_CASE("ltx2 phase log: the instrument charges its own cost to the innermost LEAF") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // (1) A SPAN THAT ENCLOSES EVERYTHING, exactly as the driver's `load` does.
  const size_t span = log.Open("unit.span", /*span=*/true);
  // (2) A BOUNDARY WITH NO LEAF LIVE. Only the span is open, so this pair is
  // charged to the TABLE and not to the span.
  const double before_gap = log.Instrument();
  { const phase::Scope gap_probe("unit.gap_probe"); }
  const double after_gap = log.Instrument();
  CHECK_MESSAGE(after_gap > before_gap,
                "opening and closing a leaf under a SPAN charged the table nothing, so either "
                "the instrument is not measuring its own boundaries or it charged them to the "
                "span. A span is not summed, so that time would vanish from the table");

  // (3) A LEAF WITH A NESTED CHILD. The child's boundaries are wall spent inside
  // the parent and outside the child.
  const size_t parent = log.Open("unit.parent", /*span=*/false);
  const double table_before_child = log.Instrument();
  for (int i = 0; i < 8; ++i) {
    const phase::Scope child("unit.child");
  }
  const double table_after_child = log.Instrument();
  // EXACTLY EQUAL, not `Approx`. `doctest::Approx` scales its epsilon by
  // `max(1, |value|)`, so on a quantity of ~1e-4 s it tolerates 1.19e-5 s —
  // 11.9 us, which is about one whole boundary. That is the size of the leak
  // this line exists to detect, so the tolerance would have been the blind spot.
  // Nothing here may charge the table AT ALL while a leaf is live, so the two
  // reads are the same double.
  CHECK_MESSAGE(table_after_child == table_before_child,
                "eight nested boundaries moved the TABLE's charge by "
                    << (table_after_child - table_before_child)
                    << "s while a leaf was live. They belong to the leaf that contains them; "
                       "charging them to the table would report the parent as enclosing a "
                       "phase nobody named");
  log.Close(parent);
  log.Close(span);

  const std::vector<phase::Record> records = log.Records();
  double parent_instrument = -1.0;
  double parent_duration = -1.0;
  double child_total = 0.0;
  int64_t children = 0;
  for (const phase::Record& r : records) {
    if (r.name == "unit.parent") {
      parent_instrument = r.instrument_seconds;
      parent_duration = r.end - r.start;
    }
    if (r.name == "unit.child") {
      child_total += r.end - r.start;
      ++children;
      CHECK_MESSAGE(r.nested, "'unit.child' opened inside a live leaf and is not marked nested");
    }
  }
  REQUIRE(children == 8);
  REQUIRE(parent_duration > 0.0);
  CHECK_MESSAGE(parent_instrument > 0.0,
                "the parent leaf was charged " << parent_instrument
                    << "s although eight children opened and closed inside it. This is the "
                       "quantity a reader of the coverage ratio subtracts");

  // AND WHAT IS **NOT** ASSERTED HERE, because a fresh review of the withdrawn
  // design measured it. This case shipped twice with a bound on
  // `uncovered / parent_instrument`, and the shipped binary reddened 2 of 200
  // consecutive runs at load 85, while a standalone probe of this exact shape
  // reddened 28 of 160 at load 125 and reached 14.1 under ASan. Decomposing the
  // parent's uncovered time explains it: fast, the inter-child gaps are 9-20 us
  // over seven boundaries against a 13-22 us charge; slow, the gaps are
  // 91-105 us against a 52-61 us charge. The UN-instrumented part of a boundary
  // — the `lock_guard` release, the `Close` return, the `Scope` destructor and
  // constructor, the call into `Open` up to its clock read — dilates faster than
  // the instrumented part. Eight bare scopes carry neither a `Tick` nor a
  // `/proc/self/statm` read inside the instrumented region, which makes this the
  // worst-conditioned probe of that ratio anywhere, not the tightest. It is
  // reported so a reader can see it move, and asserted nowhere.
  const double uncovered = parent_duration - child_total;
  MESSAGE("unit.parent = " << parent_duration << "s, children " << child_total
                           << "s, uncovered " << uncovered << "s, charged " << parent_instrument
                           << "s (ratio " << (uncovered / parent_instrument)
                           << ", REPORTED not asserted -- see the note above)");
  log.Reset();
}

// ─── the accounting is CONSERVED (#1668) ─────────────────────────────────────
//
// `instrument_seconds` at the top of the table and `instrument_seconds` on each
// record are ONE quantity split two ways, so a charge that reached neither would
// be an unmeasured cost invisible to every reader of either number. Everything
// asserted here is an inequality between two numbers in the same file:
// non-negative, no record charged more than its own duration, and the table's
// share no larger than the residue it claims to be part of. A box under load
// moves every one of these numbers and moves none of these verdicts.
TEST_CASE("ltx2 phase log: the instrument's own cost is CONSERVED across the table and its records") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();
  {
    const phase::Scope one("unit.one");
    SleepMs(30);
    { const phase::Scope inner("unit.one.inner"); }
  }
  {
    const phase::Scope two("unit.two");
    SleepMs(30);
  }

  const TableFile file;
  const nlohmann::json table = file.Write(log);

  REQUIRE(table.contains("instrument_seconds"));
  const double table_charge = table["instrument_seconds"].get<double>();
  CHECK(table_charge >= 0.0);
  double record_charge = 0.0;
  for (const nlohmann::json& e : table["phases"]) {
    REQUIRE_MESSAGE(e.contains("instrument_seconds"),
                    "the record for '" << e["name"].get<std::string>()
                                       << "' carries no instrument charge");
    const double c = e["instrument_seconds"].get<double>();
    CHECK_MESSAGE(c >= 0.0, "'" << e["name"].get<std::string>() << "' was charged " << c << "s");
    CHECK_MESSAGE(c <= e["duration_seconds"].get<double>() + 1e-9,
                  "'" << e["name"].get<std::string>() << "' was charged " << c
                      << "s of its own " << e["duration_seconds"].get<double>()
                      << "s duration, which is more instrument than record");
    record_charge += c;
  }
  MESSAGE("instrument: table " << table_charge << "s + records " << record_charge << "s");
  CHECK_MESSAGE(record_charge > 0.0,
                "no record carries any instrument charge, so the per-record half of the "
                "accounting is not reaching the emitted table");

  const double wall = table["wall_seconds"].get<double>();
  const double unaccounted = table["unaccounted_seconds"].get<double>();
  MESSAGE("wall " << wall << "s, unaccounted " << unaccounted << "s, table charge "
                  << table_charge << "s");
  REQUIRE(wall > 0.0);
  CHECK_MESSAGE(table_charge > 0.0,
                "the table's own instrument charge is " << table_charge
                    << "s across a timeline that opened and closed three scopes with nothing "
                       "live between the last two, so `ChargeLocked` never reached the `no live "
                       "leaf` arm. That arm is the whole of `unaccounted_seconds`'s explanation");
  CHECK_MESSAGE(unaccounted >= table_charge - 1e-9,
                "the table reports " << unaccounted << "s of un-named time and claims "
                    << table_charge
                    << "s of it is this instrument's own. A charge larger than the residue it "
                       "is part of means the accounting is charging intervals that are inside a "
                       "leaf to the table, which would make every residue bound too loose");
  log.Reset();
}

// ─── the residue is DECOMPOSED into the gaps that make it (#1571) ────────────
//
// `unaccounted_seconds` shipped as an aggregate, and four issues — #1439, #1470,
// #1494 and #1536 — argued about whether its 95% floor was the right tolerance
// without anyone splitting it into the gaps between consecutive leaves.
// Splitting it took one pass over the table the render already writes and
// settled the question: **92% of the residue was ONE gap**, the load's prologue
// from the timeline's origin to `Open("load.dit")`, 17.661 ms of 19.178 ms,
// while the sixteen gaps between adjacent named phases held 6.8 us each. That
// pass was a scratch script nobody shipped.
//
// THE IDENTITY IS THE GATE, and it is arithmetic rather than a tolerance. The
// leaves `Sum` adds are non-overlapping and start-ordered, so the complement of
// their union inside `[0, wall]` is exactly `wall - sum_leaf_seconds`. The gaps
// therefore add to `unaccounted_seconds` by construction, and a decomposition
// that dropped one, double counted one or mis-ordered the leaves fails by an
// amount no box load can supply.
//
// THE ONE DURATION HERE IS A LOWER BOUND ON A SLEEP, which is the only shape of
// wall-clock assertion contention cannot break: `sleep_for` returns no earlier
// than its argument and a loaded box only makes it later.
TEST_CASE("ltx2 phase log: the emitted table DECOMPOSES its residue into the gaps between leaves") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // THE PROLOGUE, which is the shape of the defect this decomposition was
  // written to find: time before the first named leaf, inside no leaf, invisible
  // in every aggregate. A span is open across it exactly as the driver's `load`
  // span is, so this gap is inside a span and outside every leaf — the position
  // `Sum` cannot see.
  const int kPrologueMs = 60;
  const phase::Scope enclosing("unit.enclosing", /*span=*/true);
  SleepMs(kPrologueMs);
  { const phase::Scope a("unit.a"); SleepMs(5); }
  SleepMs(5);  // an interior gap, between two adjacent named leaves
  {
    const phase::Scope b("unit.b");
    SleepMs(5);
    // A nested child, so the decomposition has to skip a record that is inside a
    // leaf rather than between two. Counting it would produce a negative gap.
    const phase::Scope inner("unit.b.inner");
    SleepMs(1);
  }
  SleepMs(5);  // the tail gap, before the writer reads the clock

  const TableFile file;
  const nlohmann::json table = file.Write(log);

  REQUIRE_MESSAGE(table.contains("gaps"),
                  "the emitter writes `unaccounted_seconds` as an aggregate and nothing else, "
                  "so a reader still cannot see WHICH gap holds it without writing a script -- "
                  "which is how four issues argued about a tolerance instead (#1571)");
  REQUIRE(table["gaps"].is_array());
  REQUIRE(table.contains("gap_rule"));
  CHECK(!table["gap_rule"].get<std::string>().empty());

  // The leaves the decomposition must lie between, in the emitter's own order.
  std::vector<std::string> leaf_names;
  for (const nlohmann::json& e : table["phases"]) {
    if (e.value("span", false) || e.value("nested", false)) continue;
    leaf_names.push_back(e["name"].get<std::string>());
  }
  REQUIRE(leaf_names.size() == 2);

  const nlohmann::json& gaps = table["gaps"];
  REQUIRE(!gaps.empty());

  // (1) NO GAP IS NEGATIVE, AND THEY ADD TO THE RESIDUE. Both come first and
  // neither is fatal, so one broken decomposition reports every way it is
  // broken rather than the first one. THE ORDER HERE IS A REPAIR: the count
  // below was a `REQUIRE` above this loop, and a mutation that counted NESTED
  // records as leaves aborted the case on the count and never reached these two
  // — so the negative gap that same mutation produces went unobserved, and two
  // of this case's three assertions were unproven while the case reddened.
  double gap_total = 0.0;
  for (size_t i = 0; i < gaps.size(); ++i) {
    const nlohmann::json& g = gaps[i];
    REQUIRE(g.contains("after"));
    REQUIRE(g.contains("before"));
    REQUIRE(g.contains("seconds"));
    const double seconds = g["seconds"].get<double>();
    INFO("gap " << i << " = " << g["after"].get<std::string>() << " -> "
                << g["before"].get<std::string>());
    CHECK_MESSAGE(seconds >= 0.0,
                  "gap " << i << " between '" << g["after"].get<std::string>() << "' and '"
                         << g["before"].get<std::string>() << "' is " << seconds
                         << "s. A negative gap means two records the emitter is treating as "
                            "non-overlapping leaves overlap, which would make every sum in this "
                            "table the residue of double counting");
    gap_total += seconds;
  }

  // THE IDENTITY. No tolerance beyond double rounding over a handful of
  // additions: this is the same arithmetic `Sum` does, read from the other side.
  const double unaccounted = table["unaccounted_seconds"].get<double>();
  MESSAGE("gaps sum " << gap_total << "s against an unaccounted " << unaccounted << "s over "
                      << gaps.size() << " gaps");
  CHECK_MESSAGE(std::fabs(gap_total - unaccounted) < 1e-9,
                "the gaps add to " << gap_total << "s and the table reports " << unaccounted
                    << "s of un-named time. A decomposition that does not reconcile with the "
                       "quantity it decomposes sends the next reader after the wrong region");

  // (2) ONE GAP BEFORE EACH LEAF AND ONE AFTER THE LAST. A decomposition with a
  // different count is not a partition of the timeline, whatever its sum says.
  CHECK_MESSAGE(gaps.size() == leaf_names.size() + 1,
                "the table names " << leaf_names.size() << " leaves and reports " << gaps.size()
                    << " gaps. A partition of `[0, wall]` by N non-overlapping leaves has "
                       "exactly N+1 complementary intervals");

  // (3) AND EACH GAP NAMES THE TWO LEAVES IT LIES BETWEEN, which is the half a
  // reader uses. A sum that reconciles while the names are wrong points the next
  // investigation at the wrong region, which is the failure #1571 is about.
  // Guarded on the count, because the pairing below is only defined when the
  // decomposition IS a partition — and the guard is announced rather than
  // silent, since an assertion that turned itself off would look exactly like
  // one that passed.
  if (gaps.size() != leaf_names.size() + 1) {
    MESSAGE("  the gap/leaf pairing is SKIPPED: the counts above already disagree, so there is "
            "no pairing to check. The count assertion is what speaks here.");
  } else {
    for (size_t i = 0; i < gaps.size(); ++i) {
      const std::string after = gaps[i]["after"].get<std::string>();
      const std::string before = gaps[i]["before"].get<std::string>();
      const std::string expect_after = i == 0 ? std::string("<origin>") : leaf_names[i - 1];
      const std::string expect_before =
          i == leaf_names.size() ? std::string("<end>") : leaf_names[i];
      INFO("gap " << i);
      CHECK_MESSAGE(after == expect_after,
                    "gap " << i << " says it follows '" << after
                           << "' and the table's leaf order says '" << expect_after << "'");
      CHECK_MESSAGE(before == expect_before,
                    "gap " << i << " says it precedes '" << before
                           << "' and the table's leaf order says '" << expect_before << "'");
    }
  }

  // (4) AND THE PROLOGUE IS THE ONE A READER NEEDS TO SEE. `sleep_for` returns
  // no earlier than its argument, so this lower bound is one contention can only
  // move away from red. Before this decomposition existed, exactly this region
  // was 92% of a real render's residue and no reader of the file could name it.
  const double prologue = gaps[0]["seconds"].get<double>();
  CHECK_MESSAGE(prologue >= 0.001 * static_cast<double>(kPrologueMs) - 1e-3,
                "the timeline slept " << kPrologueMs
                    << "ms inside a span and outside every leaf, and the decomposition reports "
                    << prologue
                    << "s before the first leaf. The prologue is the region that held 92% of "
                       "the LTX-2.5 load's residue, and a decomposition that cannot see it is "
                       "the aggregate it replaced");
  log.Reset();
}

// ─── the writer's clock stops BEFORE the writer works (#1569) ────────────────
//
// `PhaseLog::WriteJson` reads `Elapsed()` before it copies and sorts the record
// vector, so the writer's own serialization is not charged to `wall_seconds` and
// therefore not to `unaccounted_seconds`. That table measures the RENDER.
//
// **NOTHING ASSERTED IT, AND ITS OWN MUTATION STAYED GREEN 10 OF 10.** A fresh
// review of #1556 restored the late clock read and the case that claimed to pin
// the ordering passed every time, at `wall 0.0608987s, unaccounted 0.000534223s,
// table charge 0.000301655s`, because the copy and the sort of a THREE-record
// table are nanoseconds — far below the slack in any bound that case carried.
// An instrument whose own mutation cannot fail is not an instrument (#1569).
//
// WHAT MAKES IT GATEABLE IS A TABLE BIG ENOUGH FOR THE SORT TO EXIST, and a
// discriminator measured in the same run rather than written down as a constant.
// The case builds `kRecords` leaves, then measures two quantities K times:
//
//   * `head` — the elapsed clock read by this case immediately before the call,
//     against the `wall_seconds` the writer recorded. With the clock read first
//     the writer's clock is one function call and one uncontended mutex behind
//     this case's own, i.e. the instrument's resolution. With it read late the
//     head contains a whole copy and a whole `stable_sort`.
//   * `serialize` — that same copy and that same sort, performed by this case
//     through the same public `Records()`, on the same data, on this box, in
//     this run. It is the size of the defect, measured rather than assumed.
//
// AND THE ESTIMATOR IS A MINIMUM, WHICH IS WHY THIS IS NOT THE WITHDRAWN BOUND
// AGAIN. Contention is ONE-SIDED: it can only make a measured interval longer,
// never shorter. The honest head is a floor of ~1e-7 s plus a preemption that
// lands in it sometimes; the mutated head has a HARD floor of one serialization,
// which is present in every single iteration. A minimum over K iterations
// therefore strips the sporadic term from the honest side and cannot strip the
// deterministic term from the defective side. That is the difference between
// this and `residue <= 2 * instrument`: there, both sides were single
// measurements of comparable magnitude and the tail decided the gate; here the
// two sides differ by orders of magnitude and the estimator removes the tail by
// construction.
//
// THE FACTOR IS 0.5 AND IT IS NOT A TOLERANCE. Under the correct ordering the
// head contains ZERO copies and ZERO sorts. Under the mutated ordering it
// contains exactly one of each, so it is at least 1.0 x `serialize` by the
// definition of the two quantities. Any constant strictly between 0 and 1
// separates them; 0.5 is the midpoint, and the measured separation on this tree
// is about five orders of magnitude rather than a factor of two.
TEST_CASE("ltx2 phase log: the emitter reads its CLOCK before it serialises the table") {
  phase::PhaseLog& log = phase::PhaseLog::Instance();
  log.Reset();
  log.Begin();

  // ENOUGH RECORDS FOR THE SORT TO BE A MEASURABLE EVENT. Three records made
  // this ungateable; the sort is `n log n` on a vector of records carrying a
  // `std::string`, so the discriminator grows with `kRecords` while the honest
  // head does not depend on it at all.
  const int kRecords = 4000;
  // ONE SPAN HELD OPEN ACROSS THE BUILD, for two reasons. It stops the sampler
  // thread from being created and joined once per leaf, which would dominate the
  // build; and closing it before the measurement leaves NOTHING live, so the
  // 100 ms worker is not running while the clocks below are read.
  {
    const phase::Scope holder("unit.holder", /*span=*/true);
    for (int i = 0; i < kRecords; ++i) {
      const phase::Scope leaf("unit.leaf");
    }
  }

  const TableFile file;
  const int kProbes = 5;
  double head = -1.0;
  double serialize = -1.0;
  int64_t emitted = 0;
  for (int k = 0; k < kProbes; ++k) {
    const double before_call = log.Elapsed();
    const nlohmann::json table = file.Write(log);
    const double writer_clock = table["wall_seconds"].get<double>();
    emitted = static_cast<int64_t>(table["phases"].size());
    const double this_head = writer_clock - before_call;
    if (head < 0.0 || this_head < head) head = this_head;

    // THE DISCRIMINATOR, MEASURED THE SAME WAY THE WRITER DOES IT. `Records()`
    // returns a copy taken under the process-wide mutex and `ByStart` sorts that
    // copy; this is the same copy and the same sort through the same public
    // entry point, so it is the cost the writer would pay after its clock read
    // rather than a number quoted from another box.
    const double before_sort = log.Elapsed();
    std::vector<phase::Record> copy = log.Records();
    std::stable_sort(copy.begin(), copy.end(),
                     [](const phase::Record& a, const phase::Record& b) {
                       return a.start < b.start;
                     });
    const double this_serialize = log.Elapsed() - before_sort;
    // Kept from being optimised away: the sorted copy has to be observed.
    REQUIRE(!copy.empty());
    if (serialize < 0.0 || this_serialize < serialize) serialize = this_serialize;
  }

  REQUIRE_MESSAGE(emitted >= kRecords,
                  "the timeline was built with " << kRecords << " leaves and the table carries "
                      << emitted << " records, so the discriminator below was measured over a "
                                    "table that is not the one this case built");
  // THE INSTRUMENT'S OWN PRECONDITION, and it is what stops this case from being
  // a mute switch. If the copy and the sort cost nothing measurable, then the
  // bound below is `head < 0` and no ordering can satisfy it — but equally, a
  // `serialize` that collapsed toward the clock's resolution would make the
  // comparison meaningless in the other direction. It has to be an event.
  REQUIRE_MESSAGE(serialize > 1e-5,
                  "copying and sorting " << emitted << " records measured " << serialize
                      << "s, which is at or below this clock's own resolution. The difference "
                         "between the two orderings is that copy and that sort, so a table this "
                         "cheap to serialise cannot separate them -- which is exactly why the "
                         "three-record case in #1569 stayed green under its own mutation");
  MESSAGE("writer clock lag " << head << "s against a serialization cost of " << serialize
                              << "s over " << emitted << " records (min of " << kProbes
                              << " probes, ratio " << (head / serialize) << ")");
  CHECK_MESSAGE(head < 0.5 * serialize,
                "`WriteJson` recorded a wall " << head
                    << "s later than the clock this case read immediately before calling it, "
                       "against a measured copy-and-sort of "
                    << serialize << "s over " << emitted
                    << " records. The writer is reading its clock AFTER it serialises the "
                       "table, so its own copy and sort are charged to `wall_seconds` and "
                       "therefore to `unaccounted_seconds`. This table measures the render");
  log.Reset();
}
