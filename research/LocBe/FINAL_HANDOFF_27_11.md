# LocBe — DCTR-C3-T | Bàn giao thực nghiệm cuối (cập nhật 26/11/2026)

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

**BẢO MẬT / CHIA SẺ:** metadata Drive tại lần kiểm tra 26/11 vẫn ghi `Anyone: Writer`. **Cần đổi quyền thành `Viewer` hoặc `Restricted` + cấp xem cho nhóm** trước khi gửi link công khai hoặc mở PR; sau đó kiểm tra Quân có thể truy cập. Không tự coi đã sửa quyền.

### MFNet — chưa đưa lên Drive

| Phương pháp/seed | File chạy trên Kaggle | SHA256 do notebook ghi nhận |
|---|---|---|
| SAGE / 0 | `CS406_LOCBE_MFNET/runs/sage_b3_seed0_mfnet_9class/last.pt` | `b51087c73a2ad43a098708b827abfa68161f7280c0eb5a1656e66c147e50e356` |
| C3-T / 0 | `CS406_LOCBE_MFNET/runs/c3t_b3_seed0_mfnet_9class/last.pt` | `5554340fa5d87e144c07e24533dcf39378490399feca7da00f006083de3d1ad9` |

Hai checkpoint MFNet **không ở trong hai ZIP báo cáo đã gửi và chưa xác minh được tồn tại trên Drive**. Để lưu bản gốc, thực hiện hướng dẫn trong `HUONG_DAN_LUU_CHECKPOINT_MFNET_KAGGLE.md` đi kèm gói bàn giao ngoài repo.

## 4. Giới hạn / công việc có thể mở rộng

- MFNet chỉ mới **seed 0**, chưa có 3-seed mean±SD; chưa train TB B3 MFNet.
- MFNet Test có phần tăng +0.6066 pp, nhưng Validation giảm −0.3116 pp và chưa có CI.
- FMB qualitative chưa có prediction SegFormer B3 của baseline SAGE trên đúng ảnh để kết luận ảnh nào thắng/thua; FMB Test đã bị nhìn trước.
- FMB fusion EN/SD/SCD được đo, MS-SSIM chưa đo do chưa khóa quy ước dual-reference; runtime thành phần đã đo nhưng chưa đo end-to-end SAGE→C3-T→B3; matched C2 ablation/TTA chưa xác minh.
- **SemanticRT chưa chạy:** không trình bày kết quả MFNet như kết quả SemanticRT. MFNet là mở rộng cross-dataset riêng.
- Dữ liệu FMB/SAGE Student binary đầy đủ và 4707 ảnh fusion MFNet không nằm trọn trong snapshot; phải có nguồn Kaggle/Drive tương ứng để tái lập end-to-end.

## 5. Điều kiện chốt bàn giao

1. Sửa quyền truy cập Drive từ Anyone–Writer sang Viewer/Restricted và thử quyền truy cập của nhóm.
2. Tải checkpoint MFNet SAGE/C3-T seed0 từ Kaggle Output, kiểm SHA256, lưu Drive; điền URL vào bản bổ sung tiếp theo.
3. Commit tài liệu cuối trên `LocBe`; **không merge `main` trước review**; mở Draft PR gửi Quân.
4. Cung cấp ZIP snapshot, checksum ZIP và mô tả rõ nguồn dữ liệu / giới hạn đo; không đánh dấu mục chưa hoàn thành là `DONE`.

**Frozen code, 3 checkpoint FMB, kết quả MFNet 100 epochs seed0 và lựa chọn tham số C3-T không thay đổi trong đợt cập nhật tài liệu.**
