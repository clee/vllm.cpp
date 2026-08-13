[CmdletBinding()]
param(
    [string]$SourceDir = (Resolve-Path (Join-Path $PSScriptRoot "..")),
    [string]$BuildDir = (Join-Path $SourceDir "build-release-windows-cpu"),
    [string]$StageDir = (Join-Path $BuildDir "stage"),
    [ValidateSet("cpu", "vulkan")][string]$Backend = "cpu",
    [string]$ArtifactId = "",
    [string]$SmokeModel = (Join-Path $SourceDir "tests/vllm/models/fixtures/llama_embed_e2e"),
    [int]$SmokePort = 18080,
    [switch]$ContractTest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ArtifactId) { $ArtifactId = "windows-x86_64-msvc-$Backend" }
if ($ArtifactId -ne "windows-x86_64-msvc-$Backend") {
    throw "artifact ID must exactly match selected Windows backend"
}
function Invoke-Checked {
    param([Parameter(Mandatory)][string]$Program,
          [Parameter(Mandatory)][AllowEmptyCollection()][string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with status $LASTEXITCODE"
    }
}

function Invoke-CheckedContractTests {
    $temporaryDir = Join-Path ([System.IO.Path]::GetTempPath()) `
        "vllm-invoke-checked-$([guid]::NewGuid().ToString('N'))"
    $recordingTarget = Join-Path $temporaryDir "record-arguments.ps1"
    $failingTarget = Join-Path $temporaryDir "fail.ps1"
    $callLog = Join-Path $temporaryDir "calls.txt"
    $savedCallLog = $env:VLLM_INVOKE_CHECKED_LOG
    $powerShellExecutable = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    try {
        New-Item -ItemType Directory -Path $temporaryDir | Out-Null
        @'
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArguments = @()
)
[pscustomobject]@{
    Count = @($RemainingArguments).Count
    Arguments = @($RemainingArguments)
} | ConvertTo-Json -Compress | Set-Content `
    -LiteralPath $env:VLLM_INVOKE_CHECKED_LOG -Encoding utf8
exit 0
'@ | Set-Content -LiteralPath $recordingTarget -Encoding utf8
        @'
[string]$PID | Set-Content `
    -LiteralPath $env:VLLM_INVOKE_CHECKED_LOG -Encoding ascii
exit 23
'@ | Set-Content -LiteralPath $failingTarget -Encoding utf8
        $env:VLLM_INVOKE_CHECKED_LOG = $callLog
        $parentProcessId = $PID

        Invoke-Checked $powerShellExecutable @(
            "-NoProfile", "-NonInteractive", "-File", $recordingTarget)
        $zeroArgumentRecord = Get-Content -LiteralPath $callLog -Raw | ConvertFrom-Json
        if ([int]$zeroArgumentRecord.Count -ne 0 -or
            @($zeroArgumentRecord.Arguments).Count -ne 0) {
            throw "zero-argument target was not invoked exactly once without arguments"
        }

        Remove-Item -LiteralPath $callLog
        Invoke-Checked $powerShellExecutable @(
            "-NoProfile", "-NonInteractive", "-File", $recordingTarget,
            "alpha", "two words", "--flag=value")
        $nonemptyArgumentRecord = Get-Content -LiteralPath $callLog -Raw | ConvertFrom-Json
        if ([int]$nonemptyArgumentRecord.Count -ne 3 -or
            @($nonemptyArgumentRecord.Arguments).Count -ne 3 -or
            $nonemptyArgumentRecord.Arguments[0] -cne "alpha" -or
            $nonemptyArgumentRecord.Arguments[1] -cne "two words" -or
            $nonemptyArgumentRecord.Arguments[2] -cne "--flag=value") {
            throw "nonempty arguments did not arrive unchanged"
        }

        $rejected = $false
        try {
            Invoke-Checked $powerShellExecutable @(
                "-NoProfile", "-NonInteractive", "-File", $failingTarget)
        } catch {
            if ($_.Exception.Message -notmatch 'exited with status 23') {
                throw
            }
            $rejected = $true
        }
        if (-not $rejected) {
            throw "nonzero child exit was accepted"
        }
        $failingChildProcessId = [int](Get-Content -LiteralPath $callLog -Raw)
        Write-Host "Invoke-Checked PID contract: parent=$parentProcessId failure_child=$failingChildProcessId"
        if ($failingChildProcessId -eq $parentProcessId) {
            throw "failure target did not execute in a child process"
        }
    } finally {
        if ($null -eq $savedCallLog) {
            Remove-Item Env:VLLM_INVOKE_CHECKED_LOG -ErrorAction SilentlyContinue
        } else {
            $env:VLLM_INVOKE_CHECKED_LOG = $savedCallLog
        }
        Remove-Item -Recurse -Force $temporaryDir -ErrorAction SilentlyContinue
    }
}

function Assert-CrtPolicy {
    param([Parameter(Mandatory)][string[]]$DirectiveOutput,
          [Parameter(Mandatory)][string[]]$ImportOutput)
    $directives = $DirectiveOutput -join "`n"
    $imports = $ImportOutput -join "`n"
    if ($directives -notmatch '(?im)DEFAULTLIB\s*:\s*"?LIBCMT"?') {
        throw "COFF CRT audit: no /DEFAULTLIB:LIBCMT static CRT directive found"
    }
    if ($directives -match '(?im)DEFAULTLIB\s*:\s*"?(?:MSVCRT|MSVCPRT|LIBCMTD)"?') {
        throw "COFF CRT audit: dynamic or debug CRT directive found"
    }
    if ($imports -match '(?im)^\s*(?:VCRUNTIME[^\s]*|MSVCP[^\s]*|CONCRT[^\s]*|UCRTBASED?|api-ms-win-crt-[^\s]*|MSVCR[^\s]*)\.dll\s*$') {
        throw "PE CRT audit: dynamic or debug CRT DLL import found"
    }
}

function Invoke-CrtAudit {
    param([Parameter(Mandatory)][string[]]$Artifacts,
          [Parameter(Mandatory)][string]$Server,
          [scriptblock]$DumpbinRunner = {
              param([string]$Mode, [string]$Path)
              $output = & dumpbin $Mode $Path 2>&1
              if ($LASTEXITCODE -ne 0) {
                  throw "dumpbin $Mode failed for $Path with status $LASTEXITCODE"
              }
              return @($output)
          })
    $directiveOutput = @()
    foreach ($artifact in $Artifacts) {
        $directiveOutput += & $DumpbinRunner "/directives" $artifact
    }
    $importOutput = @(& $DumpbinRunner "/imports" $Server)
    Assert-CrtPolicy -DirectiveOutput $directiveOutput -ImportOutput $importOutput
    Write-Host ($directiveOutput -join "`n")
    Write-Host ($importOutput -join "`n")
}

function Invoke-CrtContractTests {
    $good = {
        param([string]$Mode, [string]$Path)
        if ($Mode -eq "/directives") { return '/DEFAULTLIB:"LIBCMT"' }
        return @("$Path", "KERNEL32.dll", "WS2_32.dll")
    }
    Invoke-CrtAudit -Artifacts @("fake-vllm.lib", "fake-server.obj") `
        -Server "fake-vllm-server.exe" -DumpbinRunner $good
    foreach ($bad in @(
        { param($Mode, $Path) if ($Mode -eq "/directives") { '/DEFAULTLIB:"MSVCRT"' } else { 'KERNEL32.dll' } },
        { param($Mode, $Path) if ($Mode -eq "/directives") { '/DEFAULTLIB:"LIBCMT"' } else { 'UCRTBASE.dll' } }
    )) {
        $rejected = $false
        try {
            Invoke-CrtAudit -Artifacts @("fake.lib") -Server "fake.exe" `
                -DumpbinRunner $bad
        } catch {
            $rejected = $true
        }
        if (-not $rejected) { throw "injected bad dumpbin output was accepted" }
    }
}

function Invoke-UnsupportedTierProbe {
    param([Parameter(Mandatory)][string]$TierTest,
          [scriptblock]$Runner)
    $arguments = @(
        '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran'
    )
    if ($null -eq $Runner) {
        $probeOutput = @(& $TierTest @arguments 2>&1)
        $probeExitCode = $LASTEXITCODE
    } else {
        $probeResult = & $Runner $TierTest $arguments
        $probeOutput = @($probeResult.Output)
        $probeExitCode = [int]$probeResult.ExitCode
    }
    if ($probeExitCode -ne 1) {
        throw "unsupported forced CPU tier probe exited with status $probeExitCode instead of 1"
    }
    $diagnostic = $probeOutput -join "`n"
    if ($diagnostic -notmatch [regex]::Escape("unknown x86 ISA tier 'amx'")) {
        throw "unsupported forced CPU tier probe did not report the expected diagnostic"
    }
}

function Invoke-UnsupportedTierContractTests {
    $diagnostic = "unknown x86 ISA tier 'amx'"
    $calls = [System.Collections.Generic.List[object]]::new()
    $good = {
        param([string]$Program, [string[]]$Arguments)
        $calls.Add([pscustomobject]@{
            Program = $Program
            Arguments = @($Arguments)
        }) | Out-Null
        [pscustomobject]@{ ExitCode = 1; Output = @($diagnostic) }
    }.GetNewClosure()
    Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" -Runner $good
    if ($calls.Count -ne 1) {
        throw "unsupported-tier fake runner was not invoked exactly once"
    }
    if ($calls[0].Arguments.Count -ne 1 -or
        $calls[0].Arguments[0] -ne
            '--test-case=elementwise CPU GEMM: the forced tier is the tier that actually ran') {
        throw "unsupported-tier fake runner did not receive one exact filter argument"
    }

    $badResults = @(
        [pscustomobject]@{ ExitCode = 0; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 134; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = -1073741819; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 3; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 2; Output = @($diagnostic) },
        [pscustomobject]@{ ExitCode = 1; Output = @("wrong diagnostic") }
    )
    foreach ($badResult in $badResults) {
        $runner = {
            param([string]$Program, [string[]]$Arguments)
            return $badResult
        }.GetNewClosure()
        $rejected = $false
        try {
            Invoke-UnsupportedTierProbe -TierTest "fake-tier-test.exe" `
                -Runner $runner
        } catch {
            $rejected = $true
        }
        if (-not $rejected) {
            throw "injected bad unsupported-tier result was accepted"
        }
    }
}

function Invoke-OpenAiProbeProcess {
    param([Parameter(Mandatory)][string]$Program,
          [Parameter(Mandatory)][AllowEmptyCollection()][string[]]$Arguments,
          [scriptblock]$Runner)
    if ($null -eq $Runner) {
        $output = @(& $Program @Arguments 2>&1)
        return [pscustomobject]@{
            ExitCode = [int]$LASTEXITCODE
            Output = @($output)
        }
    }
    $result = & $Runner $Program $Arguments
    if ($null -eq $result) {
        throw "OpenAI prefix probe runner returned no result"
    }
    return [pscustomobject]@{
        ExitCode = [int]$result.ExitCode
        Output = @($result.Output)
    }
}

function Invoke-OpenAiPrefixRange {
    param([Parameter(Mandatory)][string]$Program,
          [Parameter(Mandatory)][int]$First,
          [Parameter(Mandatory)][int]$Last,
          [scriptblock]$Runner)
    return Invoke-OpenAiProbeProcess -Program $Program -Arguments @(
        "--order-by=file",
        "--first=$First",
        "--last=$Last",
        "--success=true",
        "--duration=true",
        "--no-colors=true"
    ) -Runner $Runner
}

function Invoke-OpenAiPrefixBisect {
    param([Parameter(Mandatory)][string]$TestProgram,
          [scriptblock]$Runner)
    $expectedFastFailStatus = -1073740791
    $listResult = Invoke-OpenAiProbeProcess -Program $TestProgram -Arguments @(
        "--list-test-cases",
        "--order-by=file",
        "--no-version=true",
        "--no-colors=true"
    ) -Runner $Runner
    if ($listResult.ExitCode -ne 0) {
        throw "OpenAI test listing exited with status $($listResult.ExitCode)"
    }

    $testCases = [System.Collections.Generic.List[string]]::new()
    $separatorCount = 0
    foreach ($outputLine in @($listResult.Output)) {
        $line = [string]$outputLine
        if ($line -match '^=+$') {
            $separatorCount++
            if ($separatorCount -eq 2) { break }
            continue
        }
        if ($separatorCount -eq 1 -and $line.Length -gt 0) {
            $testCases.Add($line) | Out-Null
        }
    }
    if ($separatorCount -ne 2 -or $testCases.Count -lt 2) {
        throw "OpenAI test listing did not contain a complete source-order case list"
    }
    $testCount = $testCases.Count

    $fullPrefixResult = Invoke-OpenAiPrefixRange -Program $TestProgram `
        -First 1 -Last $testCount -Runner $Runner
    if ($fullPrefixResult.ExitCode -ne $expectedFastFailStatus) {
        throw "OpenAI full prefix exited with status $($fullPrefixResult.ExitCode) instead of $expectedFastFailStatus"
    }
    $shortPrefixResult = Invoke-OpenAiPrefixRange -Program $TestProgram `
        -First 1 -Last 1 -Runner $Runner
    if ($shortPrefixResult.ExitCode -ne 0) {
        throw "OpenAI first-case prefix exited with status $($shortPrefixResult.ExitCode) instead of 0"
    }

    $low = 1
    $high = $testCount
    while ($high - $low -gt 1) {
        $mid = [int][math]::Floor(($low + $high) / 2)
        $prefixResult = Invoke-OpenAiPrefixRange -Program $TestProgram `
            -First 1 -Last $mid -Runner $Runner
        if ($prefixResult.ExitCode -eq 0) {
            $low = $mid
        } elseif ($prefixResult.ExitCode -eq $expectedFastFailStatus) {
            $high = $mid
        } else {
            throw "OpenAI prefix 1..$mid exited with unexpected status $($prefixResult.ExitCode)"
        }
    }

    $firstBad = $high
    $predecessorResult = Invoke-OpenAiPrefixRange -Program $TestProgram `
        -First 1 -Last ($firstBad - 1) -Runner $Runner
    if ($predecessorResult.ExitCode -ne 0) {
        throw "OpenAI confirmed predecessor prefix exited with status $($predecessorResult.ExitCode) instead of 0"
    }
    $badPrefixResult = Invoke-OpenAiPrefixRange -Program $TestProgram `
        -First 1 -Last $firstBad -Runner $Runner
    if ($badPrefixResult.ExitCode -ne $expectedFastFailStatus) {
        throw "OpenAI confirmed bad prefix exited with status $($badPrefixResult.ExitCode) instead of $expectedFastFailStatus"
    }
    $isolatedResult = Invoke-OpenAiPrefixRange -Program $TestProgram `
        -First $firstBad -Last $firstBad -Runner $Runner
    if ($isolatedResult.ExitCode -eq 0) {
        $dependency = "cumulative"
    } elseif ($isolatedResult.ExitCode -eq $expectedFastFailStatus) {
        $dependency = "isolated"
    } else {
        throw "OpenAI isolated case $firstBad exited with unexpected status $($isolatedResult.ExitCode)"
    }

    $testName = $testCases[$firstBad - 1]
    Write-Host "OpenAI prefix bisect: first_bad=$firstBad/$testCount test=`"$testName`" predecessor_status=$($predecessorResult.ExitCode) prefix_status=$($badPrefixResult.ExitCode) isolated_status=$($isolatedResult.ExitCode) dependency=$dependency"
    return [pscustomobject]@{
        FirstBad = $firstBad
        TestCount = $testCount
        TestName = $testName
        Dependency = $dependency
    }
}

function Invoke-OpenAiPrefixBisectContractTests {
    $firstBad = 27
    $testCount = 54
    $caseNames = @(
        foreach ($index in 1..$testCount) {
            if ($index -eq $firstBad) {
                "aaa source case $index"
            } elseif ($index -eq 1) {
                "zzz source case $index"
            } else {
                "source case $index"
            }
        }
    )
    if (@($caseNames | Sort-Object)[0] -ceq $caseNames[0]) {
        throw "OpenAI prefix bisect fixture does not distinguish file order from name order"
    }
    $listing = [System.Collections.Generic.List[string]]::new()
    $listing.Add("[doctest] listing all test case names") | Out-Null
    $listing.Add("===============================================================================") | Out-Null
    foreach ($caseName in $caseNames) {
        $listing.Add($caseName) | Out-Null
    }
    $listing.Add("===============================================================================") | Out-Null
    $listing.Add("[doctest] unskipped test cases passing the current filters: $testCount") | Out-Null

    $calls = [System.Collections.Generic.List[object]]::new()
    $runner = {
        param([string]$Program, [string[]]$Arguments)
        $calls.Add([pscustomobject]@{
            Program = $Program
            Arguments = @($Arguments)
        }) | Out-Null
        if ($Arguments -contains "--list-test-cases") {
            $orderArguments = @($Arguments | Where-Object { $_ -like "--order-by=*" })
            if ($orderArguments.Count -ne 1 -or
                $orderArguments[0] -cne "--order-by=file") {
                return [pscustomobject]@{
                    ExitCode = 7
                    Output = @($caseNames | Sort-Object)
                }
            }
            return [pscustomobject]@{ ExitCode = 0; Output = @($listing) }
        }

        $firstArgument = @($Arguments | Where-Object { $_ -like "--first=*" })
        $lastArgument = @($Arguments | Where-Object { $_ -like "--last=*" })
        if ($firstArgument.Count -ne 1 -or $lastArgument.Count -ne 1 -or
            $Arguments -notcontains "--order-by=file") {
            return [pscustomobject]@{ ExitCode = 2; Output = @("bad range arguments") }
        }
        $first = [int]$firstArgument[0].Substring("--first=".Length)
        $last = [int]$lastArgument[0].Substring("--last=".Length)
        $exitCode = if ($first -eq 1 -and $last -ge $firstBad) {
            -1073740791
        } else {
            0
        }
        return [pscustomobject]@{ ExitCode = $exitCode; Output = @() }
    }.GetNewClosure()

    $captured = @(& {
        Invoke-OpenAiPrefixBisect -TestProgram "fake-openai-test.exe" `
            -Runner $runner
    } 6>&1)
    $results = @($captured | Where-Object {
        $null -ne $_.PSObject.Properties["FirstBad"]
    })
    if ($results.Count -ne 1) {
        throw "OpenAI prefix bisect did not return exactly one result"
    }
    $result = $results[0]
    if ($result.FirstBad -ne $firstBad -or
        $result.TestCount -ne $testCount -or
        $result.TestName -cne "aaa source case 27" -or
        $result.Dependency -cne "cumulative") {
        throw "OpenAI prefix bisect returned the wrong boundary"
    }
    $listCalls = @($calls | Where-Object {
        $_.Arguments -contains "--list-test-cases"
    })
    if ($listCalls.Count -ne 1) {
        throw "OpenAI prefix bisect did not list tests exactly once"
    }
    if ($listCalls[0].Arguments.Count -ne 4 -or
        $listCalls[0].Arguments[0] -cne "--list-test-cases" -or
        $listCalls[0].Arguments[1] -cne "--order-by=file" -or
        $listCalls[0].Arguments[2] -cne "--no-version=true" -or
        $listCalls[0].Arguments[3] -cne "--no-colors=true") {
        throw "OpenAI prefix bisect listing did not request exact file order"
    }
    Write-Host "OpenAI prefix bisect listing order contract OK"
    foreach ($call in @($calls | Where-Object {
        $_.Arguments -notcontains "--list-test-cases"
    })) {
        $firstArguments = @($call.Arguments | Where-Object { $_ -like "--first=*" })
        $lastArguments = @($call.Arguments | Where-Object { $_ -like "--last=*" })
        if ($firstArguments.Count -ne 1 -or $lastArguments.Count -ne 1 -or
            $call.Arguments -notcontains "--order-by=file") {
            throw "OpenAI prefix bisect emitted a non-source-order range probe"
        }
    }
    foreach ($bounds in @(
        @("--first=1", "--last=26"),
        @("--first=1", "--last=27"),
        @("--first=27", "--last=27")
    )) {
        $matches = @($calls | Where-Object {
            $_.Arguments -contains $bounds[0] -and
            $_.Arguments -contains $bounds[1]
        })
        if ($matches.Count -lt 1) {
            throw "OpenAI prefix bisect omitted confirmation $($bounds -join '..')"
        }
    }

    $midpointInjected = $false
    $unexpectedMidpointRunner = {
        param([string]$Program, [string[]]$Arguments)
        if ($Arguments -notcontains "--list-test-cases") {
            $firstArgument = @($Arguments | Where-Object { $_ -like "--first=*" })
            $lastArgument = @($Arguments | Where-Object { $_ -like "--last=*" })
            if ($firstArgument.Count -eq 1 -and $lastArgument.Count -eq 1 -and
                $firstArgument[0] -ceq "--first=1" -and
                $lastArgument[0] -ceq "--last=27" -and
                -not $midpointInjected) {
                $midpointInjected = $true
                return [pscustomobject]@{ ExitCode = 7; Output = @("injected midpoint failure") }
            }
        }
        return & $runner $Program $Arguments
    }.GetNewClosure()
    $midpointRejected = $false
    try {
        Invoke-OpenAiPrefixBisect -TestProgram "fake-openai-test.exe" `
            -Runner $unexpectedMidpointRunner | Out-Null
    } catch {
        if ($_.Exception.Message -cne
            "OpenAI prefix 1..27 exited with unexpected status 7") {
            throw
        }
        $midpointRejected = $true
    }
    if (-not $midpointRejected) {
        throw "OpenAI prefix bisect accepted an unexpected midpoint status"
    }
    Write-Host "OpenAI prefix bisect unexpected midpoint status contract OK"

    $unexpectedIsolatedRunner = {
        param([string]$Program, [string[]]$Arguments)
        if ($Arguments -notcontains "--list-test-cases" -and
            $Arguments -contains "--first=27" -and
            $Arguments -contains "--last=27") {
            return [pscustomobject]@{ ExitCode = 7; Output = @("injected isolated failure") }
        }
        return & $runner $Program $Arguments
    }.GetNewClosure()
    $isolatedRejected = $false
    try {
        Invoke-OpenAiPrefixBisect -TestProgram "fake-openai-test.exe" `
            -Runner $unexpectedIsolatedRunner | Out-Null
    } catch {
        if ($_.Exception.Message -cne
            "OpenAI isolated case 27 exited with unexpected status 7") {
            throw
        }
        $isolatedRejected = $true
    }
    if (-not $isolatedRejected) {
        throw "OpenAI prefix bisect accepted an unexpected isolated status"
    }
    Write-Host "OpenAI prefix bisect unexpected isolated status contract OK"

    $expectedDiagnostic = 'OpenAI prefix bisect: first_bad=27/54 test="aaa source case 27" predecessor_status=0 prefix_status=-1073740791 isolated_status=0 dependency=cumulative'
    $diagnostics = @($captured | Where-Object {
        ([string]$_).StartsWith("OpenAI prefix bisect: first_bad=")
    })
    if ($diagnostics.Count -ne 1 -or
        [string]$diagnostics[0] -cne $expectedDiagnostic) {
        throw "OpenAI prefix bisect emitted an unstable diagnostic schema"
    }
    Write-Host "OpenAI prefix bisect diagnostic schema contract OK"
    Write-Host $expectedDiagnostic
    Write-Host "OpenAI prefix bisect contract OK"
}

if ($ContractTest) {
    Invoke-CheckedContractTests
    Invoke-CrtContractTests
    Invoke-UnsupportedTierContractTests
    Invoke-OpenAiPrefixBisectContractTests
    Write-Host "Windows PowerShell/CRT contract tests OK"
    exit 0
}

foreach ($name in @("SOURCE_SHA", "VERSION", "EVIDENCE_URL", "SOURCE_DATE_EPOCH")) {
    if (-not [Environment]::GetEnvironmentVariable($name)) {
        throw "$name is required"
    }
}

if (-not (Test-Path (Join-Path $SmokeModel "config.json"))) {
    throw "Windows runtime smoke model is incomplete: $SmokeModel"
}

$queryDir = Join-Path $BuildDir ".cmake/api/v1/query"
New-Item -ItemType Directory -Force -Path $queryDir | Out-Null
New-Item -ItemType File -Force -Path (Join-Path $queryDir "codemodel-v2") | Out-Null

Invoke-Checked cmake @(
    "-S", $SourceDir,
    "-B", $BuildDir,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DVLLM_CPP_BUILD_TESTS=ON",
    "-DVLLM_CPP_BUILD_VERSION=$env:VERSION",
    "-DVLLM_CPP_BUILD_EXAMPLES=ON",
    "-DVLLM_CPP_SERVER=ON",
    "-DVLLM_CPP_CUDA=OFF",
    "-DVLLM_CPP_CUDA_ARCHITECTURES=",
    "-DVLLM_CPP_HIP=OFF",
    "-DVLLM_CPP_HIP_ARCHITECTURES=",
    "-DVLLM_CPP_METAL=OFF",
    "-DVLLM_CPP_MLX=OFF",
    "-DMLX_ROOT=",
    "-DVLLM_CPP_TRITON=OFF",
    "-DVLLM_CPP_VULKAN=$(if ($Backend -eq 'vulkan') { 'ON' } else { 'OFF' })",
    "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"
)
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/check-windows-portability.py"),
    "--root", $SourceDir,
    "--build-dir", $BuildDir
)

$targets = @(
    "server",
    "test_openai_api_server",
    "test_lmcache_client",
    "test_kv_offload_fs",
    "test_cpu_isa_x86",
    "test_ops_matmul_elem",
    "test_vulkan_loader"
)
if ($Backend -eq "vulkan") {
    $targets += @("test_vulkan_backend", "test_backend_cross_device")
}
Invoke-Checked cmake (@("--build", $BuildDir, "--config", "Release", "--target") + $targets)

$openaiApiServerTest = Join-Path $BuildDir "tests/Release/test_openai_api_server.exe"
Invoke-Checked $openaiApiServerTest @(
    "--test-case=api_server: socket smoke *real HTTP requests over an ephemeral port",
    "--success=true",
    "--duration=true"
)
Invoke-OpenAiPrefixBisect -TestProgram $openaiApiServerTest

foreach ($test in @(
    "test_openai_api_server.exe",
    "test_lmcache_client.exe",
    "test_kv_offload_fs.exe",
    "test_cpu_isa_x86.exe"
)) {
    Invoke-Checked (Join-Path $BuildDir "tests/Release/$test") @()
}
Invoke-Checked (Join-Path $BuildDir "tests/Release/test_vulkan_loader.exe") @()
if ($Backend -eq "vulkan") {
    Invoke-Checked (Join-Path $BuildDir "tests/Release/test_vulkan_backend.exe") @()
    Invoke-Checked (Join-Path $BuildDir "tests/Release/test_backend_cross_device.exe") @()
}

if (Test-Path $StageDir) {
    Remove-Item -Recurse -Force $StageDir
}
Invoke-Checked cmake @(
    "--install", $BuildDir,
    "--config", "Release",
    "--prefix", $StageDir,
    "--component", "vllm-server"
)

$server = Join-Path $StageDir "bin/vllm-server.exe"
if (-not (Test-Path $server)) {
    throw "native install did not stage bin/vllm-server.exe"
}
$crtArtifacts = @(
    Get-ChildItem -Path $BuildDir -Recurse -File -Include "*.obj", "vllm*.lib" |
        Where-Object { $_.FullName -notmatch '[\\/](?:_deps|third_party)[\\/]' } |
        ForEach-Object { $_.FullName }
)
if ($crtArtifacts.Count -eq 0) {
    throw "COFF CRT audit found no project objects or static libraries"
}
Invoke-CrtAudit -Artifacts $crtArtifacts -Server $server

Invoke-Checked $server @("--help")

$tierTest = Join-Path $BuildDir "tests/Release/test_ops_matmul_elem.exe"
$savedTier = $env:VT_CPU_MATMUL_TIER
try {
    $env:VT_CPU_MATMUL_TIER = "portable"
    Invoke-Checked $tierTest @()
    $env:VT_CPU_MATMUL_TIER = "avx2"
    Invoke-Checked $tierTest @()
    $env:VT_CPU_MATMUL_TIER = "amx"
    Invoke-UnsupportedTierProbe -TierTest $tierTest
} finally {
    $env:VT_CPU_MATMUL_TIER = $savedTier
}

# Python's Windows subprocess path calls CreateProcess with
# CREATE_NEW_PROCESS_GROUP, letting the smoke target one CTRL_BREAK_EVENT at the
# extracted server without broadcasting to the Actions runner's console.
$smokeHarness = Join-Path $BuildDir "windows_server_smoke.py"
@'
import json
import signal
import subprocess
import sys
import time
import urllib.request

server, model, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
proc = subprocess.Popen(
    [server, "--model", model, "--host", "127.0.0.1", "--port", str(port)],
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
)
try:
    deadline = time.monotonic() + 60
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before health check: {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    break
        except OSError:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("/health did not return 200 within 60 seconds")
        time.sleep(0.1)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/version", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"/version returned {response.status}")
        json.loads(response.read())
    proc.send_signal(signal.CTRL_BREAK_EVENT)
    if proc.wait(timeout=20) != 0:
        raise RuntimeError(f"server did not stop cleanly: {proc.returncode}")
finally:
    if proc.poll() is None:
        proc.kill()
        proc.wait()
'@ | Set-Content -LiteralPath $smokeHarness -Encoding utf8NoBOM

Invoke-Checked python @($smokeHarness, $server, $SmokeModel, "$SmokePort")

$releaseDir = Join-Path $BuildDir "release"
$metadataDir = Join-Path $releaseDir "metadata"
$tierReport = Join-Path $releaseDir "cpu-tier-report.json"
$peReport = Join-Path $releaseDir "pe-audit.json"
$archive = Join-Path $releaseDir "vllm.cpp-$($env:VERSION)-$ArtifactId.zip"
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

$passed = @{
    command = "forced Windows CPU tier"; reason = ""; result = "exit 0"
    state = "passed"; url = $env:EVIDENCE_URL
}
$absent = @{
    command = ""; reason = "not executed by the Windows preview gate"; result = ""
    state = "absent"; url = ""
}
@{
    schema = "vllm.cpp.cpu-tier-report.v1"
    selected_tier = "avx2-f16c"
    commands = @("VT_CPU_MATMUL_TIER=portable", "VT_CPU_MATMUL_TIER=avx2")
    tiers = [ordered]@{
        "portable-sse2" = $passed
        "sse2-f16c" = $absent
        "avx2-f16c" = $passed
        "avx512f" = $absent
    }
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tierReport -Encoding utf8NoBOM

$headerOutput = @(& dumpbin /nologo /headers $server 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /headers failed" }
$dependentOutput = @(& dumpbin /nologo /dependents $server 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /dependents failed" }
$rawOutput = @(& dumpbin /nologo /rawdata $server 2>&1)
if ($LASTEXITCODE -ne 0) { throw "dumpbin /rawdata failed" }
$machine = if (($headerOutput -join "`n") -match '(?im)^\s*(8664)\s+machine') { $Matches[1] } else { "" }
$imports = @(
    $dependentOutput | ForEach-Object {
        if ($_ -match '^\s*([A-Za-z0-9_.+-]+\.dll)\s*$') { $Matches[1] }
    } | Sort-Object -Unique
)
$debugPaths = @(
    (($headerOutput + $rawOutput) -join "`n") |
        Select-String -AllMatches -Pattern '(?i)[A-Za-z]:[\\/][^\r\n\x00]*?\.pdb' |
        ForEach-Object { $_.Matches.Value } | Sort-Object -Unique
)
@{
    schema = "vllm.cpp.pe-audit.v1"; machine = $machine
    imports = $imports; debug_paths = $debugPaths
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $peReport -Encoding utf8NoBOM

$compiler = (& cl 2>&1 | Select-Object -First 1) -join ""
$toolsetVersion = if ($env:VCToolsVersion) { $env:VCToolsVersion.TrimEnd('\') } else { throw "VCToolsVersion is required" }
$ucrtVersion = if ($env:UCRTVersion) { $env:UCRTVersion.TrimEnd('\') } else { throw "UCRTVersion is required" }
$abiVersion = ($toolsetVersion -split '\.')[0..1] -join '.'
$cAbiVersion = (Select-String -Path (Join-Path $SourceDir "include/vllm.h") -Pattern '^#define VLLM_ABI_VERSION ([0-9]+)$').Matches.Groups[1].Value
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/release_metadata.py"),
    "--repo-root", $SourceDir, "--build-dir", $BuildDir, "--stage-dir", $StageDir,
    "--output-dir", $metadataDir, "--tier-report", $tierReport,
    "--artifact-id", $ArtifactId, "--channel", "preview", "--backend", $Backend,
    "--version", $env:VERSION, "--c-abi-version", $cAbiVersion,
    "--source-commit", $env:SOURCE_SHA, "--source-clean", "--abi-version", $abiVersion,
    "--compiler", $compiler, "--toolchain", "Visual Studio 2022 v143 /MT",
    "--toolset-version", $toolsetVersion, "--ucrt-version", $ucrtVersion,
    "--pe-report", $peReport, "--evidence-url", $env:EVIDENCE_URL
)
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/package-server.py"), "--build-dir", $BuildDir,
    "--stage-dir", $StageDir, "--metadata-dir", $metadataDir,
    "--archive", $archive, "--archive-format", "zip", "--config", "Release"
)
$archiveExtract = Join-Path $releaseDir "archive-extracted"
if (Test-Path $archiveExtract) { Remove-Item -Recurse -Force $archiveExtract }
Expand-Archive -LiteralPath $archive -DestinationPath $archiveExtract
$archiveServer = Join-Path $archiveExtract "bin/vllm-server.exe"
if (-not (Test-Path $archiveServer)) {
    throw "final ZIP does not contain bin/vllm-server.exe"
}
Invoke-Checked $archiveServer @("--help")
Invoke-Checked python @($smokeHarness, $archiveServer, $SmokeModel, "$SmokePort")
Invoke-Checked python @(
    (Join-Path $SourceDir "scripts/validate-release-archive.py"),
    "--archive", $archive, "--archive-format", "zip",
    "--checksum", "$archive.sha256", "--provenance", "$archive.provenance.json",
    "--repo-root", $SourceDir, "--forbid-path", $BuildDir
)
Write-Host "Windows native $Backend build/stage/ZIP gate OK: $archive"
