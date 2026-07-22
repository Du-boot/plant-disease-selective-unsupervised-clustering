$ErrorActionPreference = "Stop"

$root = "F:\hospitol\G_new"
if (-not (Test-Path -LiteralPath $root)) {
    throw "Root not found: $root"
}

$classDirs = Get-ChildItem -LiteralPath $root -Directory | Sort-Object Name

for ($classIndex = 0; $classIndex -lt $classDirs.Count; $classIndex++) {
    $dir = $classDirs[$classIndex]
    $files = Get-ChildItem -LiteralPath $dir.FullName -File | Sort-Object Name
    $sampleIndex = 0

    foreach ($file in $files) {
        $dest = Join-Path $root ("{0}_{1}.jpg" -f $classIndex, $sampleIndex)
        if (Test-Path -LiteralPath $dest) {
            throw "Name collision for $dest"
        }
        Move-Item -LiteralPath $file.FullName -Destination $dest
        $sampleIndex++
    }
}

Get-ChildItem -LiteralPath $root -Directory | Remove-Item -Recurse -Force

Get-ChildItem -LiteralPath $root -File | Group-Object { $_.Name.Split('_')[0] } | Sort-Object Name | ForEach-Object {
    [PSCustomObject]@{
        ClassIndex = $_.Name
        Count      = $_.Count
    }
} | Format-Table -AutoSize
