$raw = [Console]::In.ReadToEnd()
if (-not $raw) { exit 0 }

try {
    $input_json = $raw | ConvertFrom-Json
} catch {
    exit 0
}

$cwd     = $input_json.cwd
$model   = $input_json.model.display_name
$dirname = Split-Path $cwd -Leaf

# Context remaining percentage
$remaining = $input_json.context_window.remaining_percentage
if ($null -ne $remaining -and $remaining -ne '') {
    $ctxPart = "ctx: $([math]::Round($remaining))% left"
} else {
    $ctxPart = "ctx: --"
}

# Helper: convert Unix epoch seconds to local time string, or "--" if absent
function Format-EpochTime {
    param($epoch, [string]$fmt)
    if ($null -eq $epoch -or $epoch -eq '') { return "--" }
    try {
        $dt = [System.DateTimeOffset]::FromUnixTimeSeconds([long]$epoch).LocalDateTime
        return $dt.ToString($fmt)
    } catch {
        return "--"
    }
}

# 5-hour rate limit
$fiveHourNode = $input_json.rate_limits.five_hour
if ($null -ne $fiveHourNode) {
    $fivePct   = $fiveHourNode.used_percentage
    $fiveReset = Format-EpochTime $fiveHourNode.resets_at "HH:mm"
    if ($null -ne $fivePct -and $fivePct -ne '') {
        $fivePart = "5h: $([math]::Round($fivePct))% (rst $fiveReset)"
    } else {
        $fivePart = "5h: -- (rst $fiveReset)"
    }
} else {
    $fivePart = "5h: --"
}

# 7-day rate limit
$sevenDayNode = $input_json.rate_limits.seven_day
if ($null -ne $sevenDayNode) {
    $sevenPct   = $sevenDayNode.used_percentage
    $sevenReset = Format-EpochTime $sevenDayNode.resets_at "MM/dd HH:mm"
    if ($null -ne $sevenPct -and $sevenPct -ne '') {
        $sevenPart = "7d: $([math]::Round($sevenPct))% (rst $sevenReset)"
    } else {
        $sevenPart = "7d: -- (rst $sevenReset)"
    }
} else {
    $sevenPart = "7d: --"
}

Write-Host "$dirname [$model] | $ctxPart | $fivePart | $sevenPart" 