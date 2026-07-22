$ErrorActionPreference = "Stop"

$root = "F:\hospitol\M_new_drop5"
if (-not (Test-Path -LiteralPath $root)) {
    throw "Root not found: $root"
}

$files = Get-ChildItem -LiteralPath $root -Recurse -File | Where-Object {
    $_.Extension -match '^\.(jpg|jpeg|png)$'
}

foreach ($file in $files) {
    $dest = Join-Path $root $file.Name
    if (Test-Path -LiteralPath $dest) {
        throw "Name collision for $($file.Name)"
    }
    Move-Item -LiteralPath $file.FullName -Destination $dest
}

Get-ChildItem -LiteralPath $root -Directory | Remove-Item -Recurse -Force

(Get-ChildItem -LiteralPath $root -File | Where-Object {
    $_.Extension -match '^\.(jpg|jpeg|png)$'
}).Count
