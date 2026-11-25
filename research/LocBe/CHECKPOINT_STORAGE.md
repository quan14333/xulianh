# LocBe — lưu trữ ba checkpoint SegFormer-B3 đã khóa

**Phương pháp:** DCTR-C3-T, FMB, 15 nhãn, 100 epochs, checkpoint `last.pt` theo seed.

**Thư mục Drive:** https://drive.google.com/drive/folders/1rLJSzac-jkw1M5zmk5TyVllEJGZNmBaI?usp=sharing

**LƯU Ý BẢO MẬT TRƯỚC KHI CHIA SẺ:** Google Drive metadata tại lần kiểm tra 26/11/2026 cho thấy thư mục đang đặt `anyone: writer`. Chủ sở hữu phải đổi thành **Bị hạn chế + cấp quyền Người xem cho thành viên nhóm**, hoặc tối thiểu **Anyone with link: Viewer** trước khi gửi link vào PR/public README. Bản addendum này chưa xác minh quyền đã được đổi.

| Seed | Tên file trên Drive | Google Drive file ID | Bytes | SHA256 gốc, đã xác minh local |
|---|---|---|---:|---|
| 0 | `seed0_last.pt` | `1jfJbelHD-JsKYBvhxFTju-sK9ig8HyG2` | 189162137 | `fb1a78d558a220ad1c6a105cc9724e47532d8f6a2c8230159870756c2ddf69f1` |
| 1 | `seed1_last.pt` | `1-1sYSnMvu64iFcu28okflJd3VxMrA0nW` | 189162137 | `6fdff6a36e3798c33a021eb304e36ddcea7c417e337177c1c1c917c92d74713e` |
| 2 | `seed2_last.pt` | `1jDxZvK8mlbD2svVaFsSi55IAt3icC5xi` | 189162137 | `3f6d306a77baefa70a27c2c7ac60611df8291da72c374da58dac6bc9fb4e18da` |

Các file được tải từ Kaggle trước đây dưới tên `last.zip`, `last (1).zip`, `last (2).zip`. PyTorch `torch.save` sử dụng định dạng lưu trữ ZIP nội bộ; **không cần giải nén**; đã đổi phần mở rộng và lưu trên Drive. Google Drive vẫn nhận MIME `application/x-zip-compressed` sau khi đổi tên — không đồng nghĩa là dữ liệu sai.

## Trạng thái kiểm chứng

- **Đã làm:** xác minh SHA256 của cả ba file gốc tải từ Kaggle; danh sách theo seed khớp `FROZEN_MANIFEST.json` đã khóa.
- **Đã làm:** thấy đầy đủ ba tên file và kích thước 189162137 byte trong Drive ngày 26/11/2026.
- **Chưa làm:** tải ngược từng file từ Drive để tính lại SHA256 byte-for-byte; chưa kiểm tra quyền truy cập của Quân/Mân qua tài khoản riêng.
- **Chưa làm:** MS-SSIM, end-to-end pipeline latency và matched TTA/C2 ablation. Không được đánh dấu các mục này là DONE.

## Cách kiểm tra sau khi tải từ Drive (PowerShell)

```powershell
Get-FileHash "D:\LocBe_C3T_Checkpoints\seed0_last.pt" -Algorithm SHA256
Get-FileHash "D:\LocBe_C3T_Checkpoints\seed1_last.pt" -Algorithm SHA256
Get-FileHash "D:\LocBe_C3T_Checkpoints\seed2_last.pt" -Algorithm SHA256
```

So sánh toàn bộ 64 ký tự hex SHA256 với bảng trên. Checkpoint **không được commit vào repo** (`*.pt` trong `.gitignore`, mỗi file ~189 MB); lưu ở Drive và tham chiếu bằng `FROZEN_MANIFEST.json`.

**Data protocol caveat:** kết quả FMB Test là **post-exposure / exploratory**, không phải kiểm định độc lập. Việc bổ sung liên kết Drive không thay đổi code, checkpoint hoặc mIoU.
