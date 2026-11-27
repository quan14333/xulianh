# Up branch `LocBe` bằng tài khoản có quyền GitHub

**Không nhầm:** file ZIP này **không tự tạo** branch trên máy chủ GitHub. Connected GitHub user hiện tại chỉ có quyền đọc repo `quan14333/xulianh`.

### Windows PowerShell

1. Clone repo nhóm vào ổ D, nếu bạn chưa clone:

```powershell
cd D:\
git clone https://github.com/quan14333/xulianh.git
```

2. Giải nén `METHOD_DCTR_C3T_FINAL.zip` vào thư mục riêng (ví dụ `D:\LocBe_HANDOFF`), rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File "D:\LocBe_HANDOFF\research\LocBe\scripts\PUSH_LOCBE.ps1" -Repo "D:\xulianh"
```

Script kiểm tra remote, fetch `main`, tạo/chuyển branch đúng tên **LocBe**, copy toàn bộ `research/LocBe`, commit tiếng Việt không dấu và push `origin LocBe`.

3. Mở: `https://github.com/quan14333/xulianh/tree/LocBe` và kiểm tra file `METHOD_SUMMARY.md`, `FROZEN_MANIFEST.json`, `CHECKLIST_27_11.md`.

4. Có thể tạo PR **từ LocBe → main** để Quân review; không tự merge nếu chưa được nhóm duyệt.

**Nếu lỗi HTTP 403**: nhờ Quân (chủ repo) vào Settings → Collaborators and teams và cấp quyền Write cho **tài khoản GitHub bạn dùng để push**, sau đó accept lời mời và thử lại.

**Nếu máy đã có branch LocBe**: script checkout branch đang có; tự kiểm tra trạng thái và tránh ghi đè các chỉnh sửa chưa commit của người khác.
