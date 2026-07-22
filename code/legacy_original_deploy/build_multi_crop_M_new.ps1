$ErrorActionPreference = "Stop"

$sourceRoot = "F:\hospitol\data\Multi-Crop Leaf Disease Dataset Corn, Potato, Rice\Crops leafs\Crops"
$targetRoot = "F:\hospitol\M_new"

if (-not (Test-Path -LiteralPath $sourceRoot)) {
    throw "Source root not found: $sourceRoot"
}

if (Test-Path -LiteralPath $targetRoot) {
    throw "Target already exists. Remove or rename it before rebuilding: $targetRoot"
}

New-Item -ItemType Directory -Force -Path $targetRoot | Out-Null

$classSpecs = @(
    @{ Crop = "Cashew"; Class = "Cashew leaf miner"; Display = "Cashew - leaf miner" },
    @{ Crop = "Cashew"; Class = "Cashew red rust"; Display = "Cashew - red rust" },
    @{ Crop = "Corn"; Class = "Maize leaf blight"; Display = "Corn - leaf blight" },
    @{ Crop = "Corn"; Class = "Maize streak virus"; Display = "Corn - streak virus" },
    @{ Crop = "Potato"; Class = "Fungi"; Display = "Potato - fungi" },
    @{ Crop = "Potato"; Class = "Nematode"; Display = "Potato - nematode" },
    @{ Crop = "Rice"; Class = "bacterial_leaf_blight"; Display = "Rice - bacterial leaf blight" },
    @{ Crop = "Rice"; Class = "brown_spot"; Display = "Rice - brown spot" },
    @{ Crop = "Rice"; Class = "leaf_blast"; Display = "Rice - leaf blast" },
    @{ Crop = "Tomato"; Class = "Tomato septoria leaf spot"; Display = "Tomato - septoria leaf spot" },
    @{ Crop = "Tomato"; Class = "Tomato verticulium wilt"; Display = "Tomato - verticulium wilt" }
)

$labelRows = New-Object System.Collections.Generic.List[object]
$fileRows = New-Object System.Collections.Generic.List[object]

for ($classIndex = 0; $classIndex -lt $classSpecs.Count; $classIndex++) {
    $spec = $classSpecs[$classIndex]
    $sourceDir = Join-Path (Join-Path $sourceRoot $spec.Crop) $spec.Class
    $targetDir = Join-Path $targetRoot $classIndex

    if (-not (Test-Path -LiteralPath $sourceDir)) {
        throw "Class directory not found: $sourceDir"
    }

    New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

    $files = Get-ChildItem -LiteralPath $sourceDir -File | Where-Object {
        $_.Extension -match '^\.(jpg|jpeg|png)$'
    } | Sort-Object Name

    $labelRows.Add([pscustomobject]@{
        label = $classIndex
        crop = $spec.Crop
        english_class = $spec.Class
        display_name = $spec.Display
        count = $files.Count
    })

    $i = 0
    foreach ($file in $files) {
        $newName = "{0}_{1}.jpg" -f $classIndex, $i
        $targetPath = Join-Path $targetDir $newName

        try {
            New-Item -ItemType HardLink -Path $targetPath -Target $file.FullName | Out-Null
        } catch {
            Copy-Item -LiteralPath $file.FullName -Destination $targetPath
        }

        $fileRows.Add([pscustomobject]@{
            label = $classIndex
            crop = $spec.Crop
            english_class = $spec.Class
            display_name = $spec.Display
            new_name = $newName
            original_name = $file.Name
            original_path = $file.FullName
            new_path = $targetPath
        })

        $i += 1
    }
}

$labelRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $targetRoot "label_map.csv")
$fileRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path (Join-Path $targetRoot "filename_mapping.csv")

$labelRows | Format-Table -AutoSize
Write-Host "Built renamed dataset: $targetRoot"
Write-Host "Label map: $(Join-Path $targetRoot 'label_map.csv')"
Write-Host "Filename mapping: $(Join-Path $targetRoot 'filename_mapping.csv')"
