<#
.SYNOPSIS
  Scan consent forms straight into a single PDF for the Student ID OCR app.

.DESCRIPTION
  Drives any WIA scanner already known to Windows (the HP MFP, or a Fujitsu
  once its driver is installed) with the settings this pipeline wants:
  200 DPI grayscale.

  300 DPI because that is the lowest the SP-1130N offers over WIA - it lists
  only 300 and 600. Asking for 200 is refused, and refusing it silently is
  how every page ended up cropped to two-thirds of a sheet: the resolution
  stayed at 300 while the scan extents had already been computed for 200.
  Whatever is asked for here is now checked against the device's own list.

  Scans one side only. Load the stack FACE DOWN - the printed side must face
  the rollers. Pass -Duplex to capture both sides instead, which is slower but
  survives a stack loaded the wrong way up.

  Feed only page 1 of each form - the Student ID appears nowhere else.

.EXAMPLE
  .\tools\scan.ps1
  .\tools\scan.ps1 -Flatbed -OutFile data\samples\one-form.pdf
#>
[CmdletBinding()]
param(
    [string]$OutFile = "",
    [string]$StreamDir = "",
    [string]$Scanner = "",
    [switch]$Flatbed,
    [switch]$Duplex,
    [int]$Dpi = 300,
    [ValidateSet("Grayscale", "Color")]
    [string]$Mode = "Grayscale",
    [int]$MaxPages = 100
)

$ErrorActionPreference = "Stop"

# WIA property ids and the values we care about
$WIA_HANDLING_SELECT = 3088   # 1 = feeder, 2 = flatbed
$WIA_HANDLING_STATUS = 3087   # bit 1 = paper sitting in the feeder
$WIA_DATATYPE        = 4103   # 2 = grayscale, 3 = colour
$WIA_XRES            = 6147
$WIA_YRES            = 6148
$WIA_XEXTENT         = 6151
$WIA_YEXTENT         = 6152
# JPEG rather than PNG: a third of the bytes over USB, and the OCR path
# re-encodes to JPEG anyway when it stores page images.
$WIA_FORMAT_JPEG     = "{B96B3CAE-0728-11D3-9D7B-0000F81EF32E}"

function Set-WiaProperty($Properties, [int]$Id, $Value) {
    foreach ($p in $Properties) {
        if ($p.PropertyID -eq $Id) {
            try {
                $p.Value = $Value
                return $true
            } catch {
                # Silently ignoring this is what let the scan run at one
                # resolution while the crop was sized for another.
                Write-Host ("  ! scanner refused {0} = {1}" -f $p.Name, $Value) `
                    -ForegroundColor Yellow
                return $false
            }
        }
    }
    return $false
}

function Get-AllowedValues($Properties, [int]$Id) {
    foreach ($p in $Properties) {
        if ($p.PropertyID -eq $Id) {
            try { if ($p.SubType -eq 2) { return @($p.SubTypeValues) } } catch {}
            return @()
        }
    }
    return @()
}

function Get-PropertyMax($Properties, [int]$Id) {
    foreach ($p in $Properties) {
        if ($p.PropertyID -eq $Id) {
            try { if ($p.SubType -eq 1) { return [int]$p.SubTypeMax } } catch {}
        }
    }
    return 0
}

function Get-WiaProperty($Properties, [int]$Id) {
    foreach ($p in $Properties) { if ($p.PropertyID -eq $Id) { return $p.Value } }
    return $null
}

$manager = New-Object -ComObject WIA.DeviceManager
$scanners = @($manager.DeviceInfos | Where-Object { $_.Type -eq 1 })
if ($scanners.Count -eq 0) {
    throw "No scanner found. Connect it, install its driver, and make sure Windows lists it under Printers & scanners."
}

function Get-ScannerName($DeviceInfo) {
    return ($DeviceInfo.Properties | Where-Object { $_.Name -eq "Name" }).Value
}

if ($Scanner) {
    $matched = @($scanners | Where-Object { (Get-ScannerName $_) -like "*$Scanner*" })
    if ($matched.Count -eq 0) {
        Write-Host "No scanner matching '$Scanner'. Available:" -ForegroundColor Yellow
        foreach ($s in $scanners) { Write-Host ("  " + (Get-ScannerName $s)) }
        throw "No scanner matched -Scanner '$Scanner'."
    }
    $info = $matched[0]
} elseif ($scanners.Count -eq 1) {
    $info = $scanners[0]
} else {
    # Several scanners attached - refuse to guess, because picking the wrong
    # one silently scans on the wrong machine.
    Write-Host "More than one scanner is attached. Pick one with -Scanner:" -ForegroundColor Yellow
    foreach ($s in $scanners) { Write-Host ("  -Scanner """ + (Get-ScannerName $s) + """") }
    throw "Specify -Scanner."
}

$name = Get-ScannerName $info
Write-Host "Scanner : $name"

$device = $info.Connect()
# Simplex by default: the scanner makes half as many images, which is the
# single biggest saving left in the feed. The cost is that the stack must go
# in face down - load it the wrong way and every page comes back blank, which
# the OCR reports rather than silently accepting.
#
# Use -Duplex when a mixed or uncertain stack makes that risk worse than the
# extra time; topdf.py then keeps whichever side actually has ink on it.
$duplex = (-not $Flatbed) -and $Duplex
$source = if ($Flatbed) { 2 } elseif ($duplex) { 4 } else { 1 }
Set-WiaProperty $device.Properties $WIA_HANDLING_SELECT $source | Out-Null
$sourceLabel = if ($Flatbed) { "flatbed" }
    elseif ($duplex) { "document feeder (ADF), both sides" }
    else { "document feeder (ADF), one side - load face DOWN" }
Write-Host ("Source  : {0}" -f $sourceLabel)
Write-Host "Settings: $Dpi DPI (requested), $Mode"

# Stream mode: drop each page into $StreamDir the moment it is captured so
# the caller can start reading page 1 while the scanner is still pulling
# page 2, then write a marker file when the run ends. Without this the whole
# stack has to be fed before any OCR can begin.
$streaming = [bool]$StreamDir
if ($streaming) {
    if (-not (Test-Path $StreamDir)) {
        New-Item -ItemType Directory -Path $StreamDir -Force | Out-Null
    }
}

if (-not $OutFile) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutFile = Join-Path (Join-Path $PSScriptRoot "..\data\samples") "scan-$stamp.pdf"
}
$OutFile = [System.IO.Path]::GetFullPath($OutFile)
$outDir = Split-Path $OutFile -Parent
if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }

$work = if ($streaming) { $StreamDir } else {
    Join-Path $env:TEMP ("wiascan-" + [System.Guid]::NewGuid().ToString("N"))
}
New-Item -ItemType Directory -Path $work -Force | Out-Null
$pages = @()

    # Applied once, before the loop. Re-writing resolution and extent
    # before every Transfer() makes the driver tear down and rebuild the
    # scan session on each sheet - the likeliest reason a 30ppm scanner
    # was managing one page every 15 seconds.
    $item = $device.Items.Item(1)

    # Only ask for a resolution the device lists, and fall back to its
    # lowest rather than letting a refusal pass unnoticed.
    $allowed = Get-AllowedValues $item.Properties $WIA_XRES
    $useDpi = $Dpi
    if ($allowed.Count -gt 0 -and ($allowed -notcontains $Dpi)) {
        $useDpi = ($allowed | Sort-Object)[0]
        if ($n -eq 1) {
            Write-Host ("  ! {0} DPI not supported (offers {1}); using {2}" `
                -f $Dpi, ($allowed -join "/"), $useDpi) -ForegroundColor Yellow
        }
    }
    Set-WiaProperty $item.Properties $WIA_XRES $useDpi | Out-Null
    Set-WiaProperty $item.Properties $WIA_YRES $useDpi | Out-Null

    # Trim to Letter, sized from the resolution that was actually applied
    # and clamped to what the device allows. Getting this wrong crops the
    # page, which quietly breaks every page-fraction the OCR relies on.
    $wantX = [int]($useDpi * 8.5)
    $wantY = [int]($useDpi * 11.0)
    $maxX = Get-PropertyMax $item.Properties $WIA_XEXTENT
    $maxY = Get-PropertyMax $item.Properties $WIA_YEXTENT
    if ($maxX -gt 0 -and $wantX -gt $maxX) { $wantX = $maxX }
    if ($maxY -gt 0 -and $wantY -gt $maxY) { $wantY = $maxY }
    Set-WiaProperty $item.Properties $WIA_XEXTENT $wantX | Out-Null
    Set-WiaProperty $item.Properties $WIA_YEXTENT $wantY | Out-Null
    Set-WiaProperty $item.Properties $WIA_DATATYPE `
    $(if ($Mode -eq "Color") { 3 } else { 2 }) | Out-Null

    Write-Host ("Applied  : {0} DPI, {1}x{2} px" -f $useDpi, $wantX, $wantY)

    # Per-page timings, so a slow feed can be told apart from slow saving.
    $runClock = [System.Diagnostics.Stopwatch]::StartNew()

try {
    for ($n = 1; $n -le $MaxPages; $n++) {
        $pageClock = [System.Diagnostics.Stopwatch]::StartNew()
        if (-not $Flatbed) {
            # bit 1 clear means the feeder has run dry - that is the normal
            # way a batch ends, not an error
            $status = Get-WiaProperty $device.Properties $WIA_HANDLING_STATUS
            if ($null -ne $status -and ($status -band 1) -eq 0 -and $n -gt 1) {
                Write-Host "Feeder empty - batch finished."
                break
            }
        }


        try {
            $image = $item.Transfer($WIA_FORMAT_JPEG)
        } catch {
            $msg = $_.Exception.Message
            if ($msg -match "0x80210003|out of paper|no documents|documents left") {
                if ($n -eq 1) {
                    # Nothing was ever picked up, so this is an empty tray at
                    # the start, not the end of a batch.
                    throw "The feeder is empty. Load the forms face down, top edge first, into the ADF chute at the back - push until the rollers grip. (The output tray at the front is where scanned pages land.)"
                }
                Write-Host "Feeder empty - batch finished."
                break
            }
            throw
        }

        $transferMs = $pageClock.ElapsedMilliseconds
        $pagePath = Join-Path $work ("page-{0:D3}.jpg" -f $n)
        if ($streaming) {
            # Save to a temp name and rename: the reader polls this folder,
            # and a half-written .jpg would be read as a corrupt page.
            $partial = "$pagePath.part"
            $image.SaveFile($partial)
            Move-Item -LiteralPath $partial -Destination $pagePath -Force
        } else {
            $image.SaveFile($pagePath)
        }
        $pages += $pagePath
        Write-Host ("  page {0}: transfer {1}ms, save {2}ms" -f `
            $n, $transferMs, ($pageClock.ElapsedMilliseconds - $transferMs))

        if ($Flatbed) { break }
    }

    if ($streaming) {
        Set-Content -Path (Join-Path $StreamDir "done.txt") `
            -Value ("{0}" -f $pages.Count) -Encoding ascii
        Write-Host ("Streamed {0} page(s) in {1:N1}s ({2:N1}s each)" -f `
            $pages.Count, $runClock.Elapsed.TotalSeconds, `
            ($runClock.Elapsed.TotalSeconds / [Math]::Max($pages.Count, 1)))
        return
    }

    if ($pages.Count -eq 0) { throw "Nothing was scanned. Is there paper in the feeder?" }

    # Stitch to one PDF with the project venv, so a whole stack uploads as a
    # single file rather than a pile of images.
    $python = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"))
    if (Test-Path $python) {
        $listFile = Join-Path $work "pages.txt"
        Set-Content -Path $listFile -Value ($pages -join "`n") -Encoding ascii
        $topdfArgs = @((Join-Path $PSScriptRoot "topdf.py"), $listFile, $OutFile)
        if ($duplex) { $topdfArgs += "--duplex" }
        & $python $topdfArgs
        if ($LASTEXITCODE -ne 0) { throw "PDF assembly failed." }
        Write-Host ""
        Write-Host ("Saved {0} page(s) -> {1}" -f $pages.Count, $OutFile)
        Write-Host "Upload that file at http://127.0.0.1:5000"
    } else {
        Copy-Item $pages -Destination $outDir
        Write-Host ("venv not found - left {0} image(s) in {1}" -f $pages.Count, $outDir)
    }
}
catch {
    if ($streaming) {
        Set-Content -Path (Join-Path $StreamDir "failed.txt") `
            -Value $_.Exception.Message -Encoding utf8
    }
    # A scan is expensive to repeat - if anything downstream of the capture
    # fails, keep the pages rather than making the operator feed the stack
    # through again.
    if ($pages.Count -gt 0) {
        $rescue = Join-Path $outDir ("rescued-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
        New-Item -ItemType Directory -Path $rescue -Force | Out-Null
        Copy-Item $pages -Destination $rescue -ErrorAction SilentlyContinue
        Write-Host ""
        Write-Host ("Assembly failed, but {0} scanned page(s) were kept in:" -f $pages.Count) -ForegroundColor Yellow
        Write-Host ("  {0}" -f $rescue) -ForegroundColor Yellow
        Write-Host "You can upload those images directly - the app accepts them." -ForegroundColor Yellow
    }
    throw
}
finally {
    if (-not $streaming) {
        Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}
