<#
.SYNOPSIS
  Report every scanner Windows can see, and whether scan.ps1 can drive it.

.DESCRIPTION
  Run this after connecting the Fujitsu. It names the model, says whether
  the WIA interface scan.ps1 needs is present, and reports feeder/duplex
  support and the resolutions on offer.
#>
$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "=== Scanner-ish hardware Windows has present ===" -ForegroundColor Cyan
$pnp = Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Class -match 'Image|Scanner' -or
        $_.FriendlyName -match 'fujitsu|scansnap|PFU|fi-\d|SP-\d|ix\d{3,4}|ricoh|scanner'
    }
if ($pnp) {
    $pnp | Select-Object Status, Class, FriendlyName | Format-Table -AutoSize
} else {
    Write-Host "  none found" -ForegroundColor Yellow
}

Write-Host "=== WIA scanners (what tools\scan.ps1 drives) ===" -ForegroundColor Cyan
try {
    $manager = New-Object -ComObject WIA.DeviceManager
    $scanners = @($manager.DeviceInfos | Where-Object { $_.Type -eq 1 })
    if ($scanners.Count -eq 0) {
        Write-Host "  none - scan.ps1 has nothing to talk to yet" -ForegroundColor Yellow
    }
    foreach ($info in $scanners) {
        $props = @{}
        foreach ($p in $info.Properties) { $props[$p.Name] = $p.Value }
        Write-Host ""
        Write-Host ("  Name  : {0}" -f $props['Name']) -ForegroundColor Green
        Write-Host ("  Model : {0}" -f $props['Description'])

        try {
            $device = $info.Connect()
            $caps = ($device.Properties | Where-Object { $_.PropertyID -eq 3086 }).Value
            if ($null -ne $caps) {
                $features = @()
                if ($caps -band 1) { $features += "ADF feeder" }
                if ($caps -band 2) { $features += "flatbed" }
                if ($caps -band 4) { $features += "duplex" }
                Write-Host ("  Can do: {0}" -f ($features -join ", "))
            }
            $item = $device.Items.Item(1)
            $xres = ($item.Properties | Where-Object { $_.PropertyID -eq 6147 })
            if ($xres) {
                Write-Host ("  DPI   : currently {0}" -f $xres.Value)
            }
            Write-Host "  STATUS: usable by tools\scan.ps1" -ForegroundColor Green
        } catch {
            Write-Host ("  STATUS: found but would not connect - {0}" -f $_.Exception.Message) -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host ("  WIA unavailable: {0}" -f $_.Exception.Message) -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Fujitsu / PFU software installed ===" -ForegroundColor Cyan
$paths = @(
    'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
    'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'
)
$apps = foreach ($path in $paths) {
    try { Get-ItemProperty $path -ErrorAction Stop } catch {}
}
$found = $apps |
    Where-Object { $_.DisplayName -match 'ScanSnap|Fujitsu|PFU|PaperStream|Ricoh' } |
    Select-Object -ExpandProperty DisplayName -Unique
if ($found) { $found | ForEach-Object { Write-Host "  $_" } }
else { Write-Host "  none - driver not installed yet" -ForegroundColor Yellow }

Write-Host ""
