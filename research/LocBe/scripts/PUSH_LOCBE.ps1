param([Parameter(Mandatory=$true)][string]$Repo)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path $Repo).Path
$origin = git -C $Repo remote get-url origin
if ($LASTEXITCODE -ne 0 -or $origin -notmatch 'quan14333/xulianh') { throw "Repo origin is not quan14333/xulianh: $origin" }
git -C $Repo fetch origin main
if ($LASTEXITCODE -ne 0) {throw 'git fetch failed'}
git -C $Repo switch main
if ($LASTEXITCODE -ne 0) {throw 'Cannot switch main; check uncommitted changes'}
git -C $Repo pull --ff-only origin main
if ($LASTEXITCODE -ne 0) {throw 'Cannot fast-forward main'}
$branches = git -C $Repo branch --list LocBe
if (-not $branches) { git -C $Repo switch -c LocBe } else { git -C $Repo switch LocBe }
if ($LASTEXITCODE -ne 0) {throw 'Cannot switch/create LocBe'}
$target = Join-Path $Repo 'research/LocBe'
New-Item -ItemType Directory -Force -Path $target | Out-Null
Copy-Item (Join-Path $PSScriptRoot '..\*') -Destination $target -Recurse -Force -Exclude '.git'
git -C $Repo add research/LocBe
git -C $Repo commit -m 'docs: chot phuong phap va ban giao LocBe DCTR-C3T'
if ($LASTEXITCODE -ne 0) {Write-Host 'No changes committed or commit failed: inspect git status'}
git -C $Repo push -u origin LocBe
if ($LASTEXITCODE -ne 0) {throw 'Push failed. Ask Quân to add collaborator write access or push with authorized GitHub login.'}
Write-Host "PUSHED: https://github.com/quan14333/xulianh/tree/LocBe"
