# Git workflow (Windows PowerShell)

## Clone và thiết lập

```powershell
git clone https://github.com/quan14333/xulianh.git
Set-Location xulianh
git remote -v
git status
python -m pip install -r requirements-dev.txt
python scripts/validate_repo.py
python -m pytest
```

Chủ repository cần mời đúng GitHub username của Quân, Mân, Lộc bé và Lộc lớn. Không giả định
display name chính là username và không gửi token qua chat/commit.

## Branch ban đầu

Mỗi người chạy đúng một lệnh tương ứng:

```powershell
git switch -c feature/quan-baseline
git switch -c feature/man-dataset
git switch -c feature/locbe-research
git switch -c feature/loclon-evaluation
```

Chỉ chạy lệnh của mình. Với task mới, cập nhật `main` rồi tạo branch mới:

```powershell
git switch main
git pull --ff-only origin main
git switch -c feature/ten-ngan-gon
```

## Làm việc, commit và push

```powershell
git status
git diff
python scripts/validate_repo.py
python -m pytest
git add <duong-dan-cu-the>
git diff --cached
git commit -m "feat: mo ta thay doi"
git push -u origin <ten-branch>
```

Không dùng `git add .` trước khi kiểm tra data, checkpoint, notebook output và secret. Không
force-push branch đang được người khác review nếu chưa thống nhất.

## Pull Request và review

1. Mở PR từ branch cá nhân vào `main` trên GitHub.
2. Điền đủ template, liên kết issue/card, ghi lệnh và kết quả test thật.
3. Mời ít nhất một người khác; nếu sửa module có owner khác thì owner đó phải review.
4. Chờ CI xanh và xử lý comment. Merge theo chính sách repository; không bypass review.
5. Sau merge, cập nhật local bằng `git switch main` và `git pull --ff-only origin main`.

## Đồng bộ branch đang làm

Ưu tiên merge `main` để người mới dễ hiểu và tránh rewrite lịch sử:

```powershell
git fetch origin
git switch <ten-branch>
git merge origin/main
```

Nếu conflict, Git đánh dấu `<<<<<<<`, `=======`, `>>>>>>>`. Mở từng file, hiểu cả hai phía,
xóa marker và giữ nội dung đúng; không chọn “ours/theirs” hàng loạt khi chưa đọc.

```powershell
git status
git add <file-da-giai-quyet>
git commit
python scripts/validate_repo.py
python -m pytest
git push
```

Muốn hủy merge chưa commit: `git merge --abort`. Không dùng `git reset --hard` để xử lý nhanh
vì có thể làm mất thay đổi chưa commit.

## Commit message gợi ý

- `feat: add verified FMB manifest loader`
- `fix: prevent uint8 overflow in SCD input`
- `test: add SCD reference cases`
- `docs: record preprocessing audit`
- `exp: register robustness run configuration`
