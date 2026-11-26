# LocBe — DCTR-C3-T | Bàn giao thực nghiệm cuối (cập nhật lưu trữ 27/11/2026)

**Repo/branch:** https://github.com/quan14333/xulianh/tree/LocBe/research/LocBe  
**Mốc code/evidence đã xác nhận trên GitHub:** `bb53a374d3c7cf2f94918c715b50ebfdf1065a24` (MFNet seed0); **đây không phải commit SHA của tài liệu bàn giao cuối**.  
**Phương pháp:** DCTR-C2 VIS residual + thermal-only revert-to-SAGE guard, ngưỡng cố định 0.30, không chỉnh sau khi đã xem FMB Test.

## 1. Kết quả đo thật — không trộn giao thức

| Dataset / tập | Đơn vị | SAGE | C3-T | C3-T − SAGE | Trạng thái |
|---|---|---:|---:|---:|---|
| **FMB** Test clean 273 ảnh, **3 seed**, mIoU13 | % | 63.37 | 63.1994 | −0.1706 pp | **POST-EXPOSURE / EXPLORATORY**, không phải held-out độc lập |
| **MFNet** Validation 392 ảnh, **seed 0**, mIoU9 | % | 57.5254 | 57.2138 | −0.3116 pp | Đã huấn luyện B3 9 lớp riêng trên MFNet |
| **MFNet** Test 393 ảnh, **seed 0**, mIoU9 | % | 53.4829 | 54.0895 | +0.6066 pp | Kết quả một seed, chưa có CI hoặc 3 seed |
| **MFNet** Test 393 ảnh, **seed 0**, mIoU8 bỏ GT unlabeled | % | 53.2687 | 54.2545 | +0.9858 pp | Quy ước loại pixel GT0 trước khi lập confusion |

FMB mIoU13 **không so trực tiếp** với MFNet mIoU9/mIoU8. MFNet Test có cải thiện nhỏ nhưng Val giảm; **không tuyên bố C3-T tổng quát hóa tốt hơn SAGE một cách ổn định hoặc có ý nghĩa thống kê**. MFNet dùng 784 Train **gốc** + 392 Val + 393 Test; bản phân phối `train.txt` có thêm 784 `_flip`, đã loại bản nhân đôi khỏi danh sách Train hữu hiệu (không tự chia lại). Các split không trùng base ID.

## 2. Bằng chứng đã lưu trong branch LocBe

- Phương pháp, công thức, file tham khảo: `METHOD_SUMMARY.md`, `source/`, `configs/`; freeze source C2 SHA256 `7641e319a714fd44b246e3921404ece1110bc6766dd4d5bdc1c9c35c4165d621` và C3-T SHA256 `6eda7aabead6207d54107478edc52792538383f86913dc4dd838c8879504d4b6`.
- FMB: `results/results_overall.csv`, `results/results_per_class.csv`, `evidence/inference_only/report/`, `evidence/inference_only/predictions/c3t_seed0/` (280 masks), `figures/qualitative/` (6 panel).
- MFNet: `MFNET_RESULTS_26_11.md`, `evidence/mfnet_seed0/report/`, `evidence/mfnet_seed0/runs/` (4 ma trận confusion, 2 lịch sử train, mask mẫu) và `evidence/mfnet_seed0/figures/` (6 panel). Fusion manifest liệt kê 4707 ảnh; **không có đầy đủ ảnh fusion nhị phân trong ZIP**.
- `FROZEN_MANIFEST.json` cùng `SHA256.txt`: manifest/checksum có thể rà soát; trong snapshot ZIP kèm theo có cả file raw evidence vốn có thể bị `.gitignore` loại khỏi GitHub.

## 3. Checkpoint — phân biệt rõ FMB/MFNet

### FMB — đã lưu trên Drive, chưa tải ngược để so hash

Thư mục: https://drive.google.com/drive/folders/1rLJSzac-jkw1M5zmk5TyVllEJGZNmBaI?usp=sharing

| Seed | Checkpoint FMB | SHA256 bản gốc trên máy đã xác minh |
|---|---|---|
| 0 | `seed0_last.pt` | `fb1a78d558a220ad1c6a105cc9724e47532d8f6a2c8230159870756c2ddf69f1` |
| 1 | `seed1_last.pt` | `6fdff6a36e3798c33a021eb304e36ddcea7c417e337177c1c1c917c92d74713e` |
| 2 | `seed2_last.pt` | `3f6d306a77baefa70a27c2c7ac60611df8291da72c374da58dac6bc9fb4e18da` |

**Quyền chia sẻ FMB:** metadata folder ngày 27/11 vẫn ghi `Anyone: Writer`, có rủi ro chỉnh sửa ngoài ý muốn. Quyền được giữ nguyên theo quyết định của chủ sở hữu; không tự sửa và không ghi nhận là đã an toàn. Chưa xác minh truy cập bằng tài khoản riêng của Quân/Mân.

### MFNet — đã lưu trên Drive; chưa tải ngược để tính SHA256

Folder: https://drive.google.com/drive/u/0/folders/13CNqHSPiLUgLP8pxq6ykR2jnxr-T0I1Y

| Phương pháp / seed | Tên file trên Drive | Drive file ID | Bytes | SHA256 bản gốc đã kiểm tra | File |
|---|---|---|---:|---|---|
| SAGE / seed 0 | `sage_b3_seed0_mfnet_last.pt` | `1uJvT7Oh4shsfYQPjHmlmQ5XPpenDBcbA` | 189143705 | `b51087c73a2ad43a098708b827abfa68161f7280c0eb5a1656e66c147e50e356` | [Drive](https://drive.google.com/file/d/1uJvT7Oh4shsfYQPjHmlmQ5XPpenDBcbA/view) |
| C3-T / seed 0 | `c3t_b3_seed0_mfnet_last.pt` | `1HjKgMYMTgrHLjUecBzU5ZkAACtVnICQN` | 189143705 | `5554340fa5d87e144c07e24533dcf39378490399feca7da00f006083de3d1ad9` | [Drive](https://drive.google.com/file/d/1HjKgMYMTgrHLjUecBzU5ZkAACtVnICQN/view) |

**Phạm vi xác minh:** SHA256 trên là hash **đã kiểm tra trên bản gốc**, khớp hồ sơ thí nghiệm và ghi chú người dùng cung cấp. Drive hiện chỉ được xác nhận **tên và dung lượng**; chưa tải ngược hai checkpoint để tính lại SHA256. **Không ghi nhận remote SHA256 verified** hoặc suy ra nội dung byte-for-byte từ tên/dung lượng.

**Quyền chia sẻ (27/11/2026):** metadata folder MFNet và cả hai file hiện ghi `Anyone: Writer`. Người có liên kết có thể chỉnh sửa ngoài ý muốn; quyền được **giữ nguyên theo quyết định của chủ sở hữu**, không tự đổi và không mô tả là đã an toàn. Chưa xác minh truy cập bằng tài khoản riêng của Quân/Mân.

Bản gốc từ hai run `CS406_LOCBE_MFNET/runs/sage_b3_seed0_mfnet_9class/last.pt` và `CS406_LOCBE_MFNET/runs/c3t_b3_seed0_mfnet_9class/last.pt`. Các binary vẫn không nằm trong hai ZIP báo cáo hoặc Git; MFNet 9 lớp không thay thế checkpoint FMB 15 lớp. Xem `CHECKPOINT_STORAGE.md` cho hướng dẫn đối chiếu bản tải về.

## 4. Giới hạn / công việc có thể mở rộng

- MFNet chỉ mới **seed 0**, chưa có 3-seed mean±SD; chưa train TB B3 MFNet.
- MFNet Test có phần tăng +0.6066 pp, nhưng Validation giảm −0.3116 pp và chưa có CI.
- FMB qualitative chưa có prediction SegFormer B3 của baseline SAGE trên đúng ảnh để kết luận ảnh nào thắng/thua; FMB Test đã bị nhìn trước.
- FMB fusion EN/SD/SCD được đo, MS-SSIM chưa đo do chưa khóa quy ước dual-reference; runtime thành phần đã đo nhưng chưa đo end-to-end SAGE→C3-T→B3; matched C2 ablation/TTA chưa xác minh.
- **SemanticRT chưa chạy:** không trình bày kết quả MFNet như kết quả SemanticRT. MFNet là mở rộng cross-dataset riêng.
- Dữ liệu FMB/SAGE Student binary đầy đủ và 4707 ảnh fusion MFNet không nằm trọn trong snapshot; phải có nguồn Kaggle/Drive tương ứng để tái lập end-to-end.

## 5. Điều kiện chốt bàn giao

1. Giữ nguyên quyền Drive theo quyết định của chủ sở hữu; ghi nhận `Anyone: Writer` và rủi ro chỉnh sửa ngoài ý muốn. Kiểm tra truy cập bằng tài khoản nhóm còn thiếu, không tự đánh dấu đã an toàn.
2. Hai checkpoint MFNet SAGE/C3-T seed0 đã lưu Drive, tên/dung lượng đã xác nhận và SHA256 bản gốc đã kiểm tra. Còn thiếu tải ngược từ Drive để tính lại hash; mục này chưa DONE.
3. Cập nhật tài liệu/checksum trên `LocBe`, base đã kiểm tra `cb035dbac414d1335061e1e89b85a208c23a0189`; commit/push theo yêu cầu. Không merge `main` hoặc tự tạo PR trong lượt này; review/PR do chủ sở hữu quyết định riêng.
4. Cung cấp ZIP snapshot, checksum ZIP và mô tả rõ nguồn dữ liệu / giới hạn đo; không đánh dấu mục chưa hoàn thành là `DONE`.

**Frozen code, 3 checkpoint FMB, kết quả MFNet 100 epochs seed0 và lựa chọn tham số C3-T không thay đổi trong đợt cập nhật tài liệu.**
