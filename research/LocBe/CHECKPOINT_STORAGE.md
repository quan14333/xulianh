# LocBe — lưu trữ checkpoint SegFormer-B3 FMB và MFNet

## FMB — ba checkpoint đã khóa

**Phương pháp:** DCTR-C3-T, FMB, 15 nhãn, 100 epochs, checkpoint `last.pt` theo seed.

**Thư mục Drive:** https://drive.google.com/drive/folders/1rLJSzac-jkw1M5zmk5TyVllEJGZNmBaI?usp=sharing

**Quyền chia sẻ FMB:** metadata folder tại lần kiểm tra 27/11/2026 vẫn ghi `Anyone: Writer`. Có rủi ro chỉnh sửa ngoài ý muốn. Giữ nguyên quyền theo quyết định của chủ sở hữu; bản cập nhật này không sửa quyền, không mô tả là đã an toàn và chưa xác minh truy cập bằng tài khoản riêng của Quân/Mân.

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

## MFNet — hai checkpoint seed 0 đã lưu trên Drive

**Folder:** https://drive.google.com/drive/u/0/folders/13CNqHSPiLUgLP8pxq6ykR2jnxr-T0I1Y

| Phương pháp / seed | Tên file trên Drive | Drive file ID | Bytes | SHA256 bản gốc đã kiểm tra | File |
|---|---|---|---:|---|---|
| SAGE / seed 0 | `sage_b3_seed0_mfnet_last.pt` | `1uJvT7Oh4shsfYQPjHmlmQ5XPpenDBcbA` | 189143705 | `b51087c73a2ad43a098708b827abfa68161f7280c0eb5a1656e66c147e50e356` | [Drive](https://drive.google.com/file/d/1uJvT7Oh4shsfYQPjHmlmQ5XPpenDBcbA/view) |
| C3-T / seed 0 | `c3t_b3_seed0_mfnet_last.pt` | `1HjKgMYMTgrHLjUecBzU5ZkAACtVnICQN` | 189143705 | `5554340fa5d87e144c07e24533dcf39378490399feca7da00f006083de3d1ad9` | [Drive](https://drive.google.com/file/d/1HjKgMYMTgrHLjUecBzU5ZkAACtVnICQN/view) |

**Phạm vi xác minh:** SHA256 trên là hash **đã kiểm tra trên bản gốc**, khớp hồ sơ thí nghiệm và ghi chú người dùng cung cấp. Drive hiện chỉ được xác nhận **tên và dung lượng**; chưa tải ngược hai checkpoint để tính lại SHA256. **Không ghi nhận remote SHA256 verified** hoặc suy ra nội dung byte-for-byte từ tên/dung lượng.

**Quyền chia sẻ (27/11/2026):** metadata folder MFNet và cả hai file hiện ghi `Anyone: Writer`. Người có liên kết có thể chỉnh sửa ngoài ý muốn; quyền được **giữ nguyên theo quyết định của chủ sở hữu**, không tự đổi và không mô tả là đã an toàn. Chưa xác minh truy cập bằng tài khoản riêng của Quân/Mân.

Đây là **SegFormer-B3 9 lớp MFNet**, khác ba checkpoint **15 lớp FMB** ở trên. Nguồn bản gốc: `CS406_LOCBE_MFNET/runs/sage_b3_seed0_mfnet_9class/last.pt` và `CS406_LOCBE_MFNET/runs/c3t_b3_seed0_mfnet_9class/last.pt`. Binary không nằm trong ZIP báo cáo và không được commit vào Git. Upload Drive chỉ bổ sung nơi lưu, không thay checkpoint hoặc kết quả.

Google Drive báo MIME `application/x-zip-compressed`; đây có thể là container ZIP nội bộ của `torch.save`, không cần giải nén. MIME/tên/dung lượng không thay thế kiểm tra hash.

Sau khi người nhận tải bản Drive về (chưa thực hiện trong lượt này), đối chiếu đủ 64 ký tự:

```powershell
Get-FileHash "D:\MFNet_Checkpoints\sage_b3_seed0_mfnet_last.pt" -Algorithm SHA256
Get-FileHash "D:\MFNet_Checkpoints\c3t_b3_seed0_mfnet_last.pt" -Algorithm SHA256
```

**Còn thiếu:** SHA256 tính lại từ bản tải Drive và kiểm tra truy cập bằng tài khoản của thành viên nhóm. Các giới hạn MS-SSIM, end-to-end latency, matched ablation và kết quả một seed giữ nguyên.
