$ErrorActionPreference = "Stop"

$sourceRoot = "F:\hospitol\M_new_drop5_drop7"
$targetRoot = "F:\hospitol\M_new_drop3_7"

if (-not (Test-Path -LiteralPath $sourceRoot)) {
    throw "Source root not found: $sourceRoot"
}

if (Test-Path -LiteralPath $targetRoot) {
    throw "Target already exists. Remove or rename it before rebuilding: $targetRoot"
}

New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$classSpecs = @(
    @{ Old = 0; New = 0; Crop = "Cashew"; Display = "Cashew - leaf miner"; Count = 555 },
    @{ Old = 1; New = 1; Crop = "Cashew"; Display = "Cashew - red rust"; Count = 566 },
    @{ Old = 2; New = 2; Crop = "Corn"; Display = "Corn - leaf blight"; Count = 493 },
    @{ Old = 4; New = 3; Crop = "Potato"; Display = "Potato - fungi"; Count = 390 },
    @{ Old = 5; New = 4; Crop = "Rice"; Display = "Rice - bacterial leaf blight"; Count = 401 },
    @{ Old = 6; New = 5; Crop = "Rice"; Display = "Rice - brown spot"; Count = 370 },
    @{ Old = 8; New = 6; Crop = "Tomato"; Display = "Tomato - verticulium wilt"; Count = 673 }
)

$labelRows = New-Object System.Collections.Generic.List[object]

foreach ($spec in $classSpecs) {
    $files = Get-ChildItem -LiteralPath $sourceRoot -File | Where-Object {
        $_.Name -match ("^{0}_" -f $spec.Old)
    } | Sort-Object Name

    if ($files.Count -ne $spec.Count) {
        throw "Count mismatch for old label $($spec.Old): expected $($spec.Count), got $($files.Count)"
    }

    for ($i = 0; $i -lt $files.Count; $i++) {
        $newName = "{0}_{1}.jpg" -f $spec.New, $i
        $targetPath = Join-Path $targetRoot $newName
        Copy-Item -LiteralPath $files[$i].FullName -Destination $targetPath
    }

    $labelRows.Add([pscustomobject]@{
        label = $spec.New
        old_label = $spec.Old
        crop = $spec.Crop
        display_name = $spec.Display
        count = $spec.Count
    })
}

$labelRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $targetRoot "label_map.csv")
$labelRows | Format-Table -AutoSize
Write-Host "Built renamed dataset: $targetRoot"
Write-Host "Label map: $(Join-Path $targetRoot 'label_map.csv')"
