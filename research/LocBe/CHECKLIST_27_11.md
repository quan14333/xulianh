# Checklist bàn giao LocBe — cập nhật 27/11/2026

| Mục | Trạng thái chính xác | Vị trí / ghi chú |
|---|---|---|
| Method card, 3 references | **DONE** | `METHOD_SUMMARY.md` |
| Code/config C2→C3-T frozen | **DONE** | `source/`, `configs/`, SHA256 trong `FROZEN_MANIFEST.json`, không thay công thức |
| FMB Test 273 clean/280 full, 3 seed + SAGE/TB | **DONE / POST-EXPOSURE EXPLORATORY** | `results/`, `RESULTS_DISCUSSION.md`; không gọi independent Test |
| Self-test, confusion parity | **DONE** | `evidence/inference_only/report/confusion_parity.json` PASS 280/280 seed0 |
| 280 prediction C3-T seed0, 6 panel FMB | **DONE ON GITHUB** | `evidence/inference_only/predictions/`, `figures/qualitative/` (commit `49bdafd`) |
| FMB SAGE B3 prediction trên cùng panel | **NOT DONE** | Không kết luận qualitative thắng/thua SAGE theo ảnh |
| Fusion EN/SD/SCD, component latency | **DONE / LIMITED SCOPE** | `results/fusion_metrics_summary.csv`, `results/runtime_*.json`; không có end-to-end latency |
| Fusion MS-SSIM, matched C2 ablation, TTA | **NOT DONE / OPTIONAL IF REQUIRED** | Chưa khóa protocol tương ứng |
| 3 FMB B3 checkpoint | **DRIVE STORED / LOCAL SHA VERIFIED** | `CHECKPOINT_STORAGE.md`; 3 file ~189 MB; Drive redownload SHA chưa kiểm |
| Quyền thư mục checkpoint FMB | **UNCHANGED / RISK NOTED** | 27/11 metadata vẫn `Anyone: Writer`; giữ nguyên theo quyết định chủ sở hữu, có rủi ro chỉnh sửa ngoài ý muốn; chưa xác minh tài khoản nhóm |
| Train/Dev/Test manifest / provenance FMB | **PARTIAL** | ID/hash lưu; dataset gốc & SAGE Student binary không có trọn trong snapshot |
| MFNet cross-dataset split audit | **DONE** | 784 Train original +392 Val +393 Test, `_flip` excluded; no base-ID overlap |
| MFNet fusion SAGE/TB/C3-T, metrics, 6 panel | **DONE** | `evidence/mfnet_seed0/` — 4707 manifest rows, chưa lưu tất cả PNG binary |
| MFNet SegFormer-B3 9 lớp SAGE vs C3-T | **DONE SEED0 ONLY** | 100 epochs; Test 53.4829 vs 54.0895 mIoU9; Val 57.5254 vs 57.2138 |
| MFNet SAGE + C3-T seed0 | **DRIVE STORED / ORIGINAL SHA VERIFIED** | `sage_b3_seed0_mfnet_last.pt`, `c3t_b3_seed0_mfnet_last.pt`, mỗi file 189143705 bytes; [folder](https://drive.google.com/drive/u/0/folders/13CNqHSPiLUgLP8pxq6ykR2jnxr-T0I1Y); file IDs/SHA256 trong `CHECKPOINT_STORAGE.md` |
| MFNet Drive redownload SHA256 | **NOT DONE** | Chỉ xác nhận tên/dung lượng trên Drive; SHA256 đã kiểm tra thuộc bản gốc, không ghi remote SHA256 verified |
| Quyền MFNet folder và hai checkpoint | **UNCHANGED / RISK NOTED** | Metadata 27/11 `Anyone: Writer`; không tự sửa quyền, không mô tả đã an toàn; chưa kiểm tài khoản Quân/Mân |
| MFNet full 3 seed, TB B3 baseline | **OPTIONAL / NOT DONE** | Không suy diễn significant/robust gain từ seed0 |
| SemanticRT camera-held-out | **NOT DONE** | Là dataset khác MFNet, cần thí nghiệm riêng nếu nhóm yêu cầu |
| Git branch LocBe | **VERIFIED BASE `cb035db`** | Hồ sơ trước cập nhật Drive: `cb035dbac414d1335061e1e89b85a208c23a0189`; mốc research evidence `bb53a37` giữ nguyên |
| Final handoff note/checksum package | **UPDATED FOR MFNET DRIVE METADATA** | `FINAL_HANDOFF_27_11.md`, `FROZEN_MANIFEST.json`, `SHA256.txt`; hash tài liệu và manifest cập nhật, không checksum tự tham chiếu; checkpoint Drive redownload chưa kiểm |
| Draft PR to main | **NOT CREATED IN THIS UPDATE** | Không tự tạo PR/merge trong lượt này; chủ sở hữu quyết định review/PR riêng, giữ nguyên quyền Drive |

**Đọc `FINAL_HANDOFF_27_11.md` để biết toàn bộ giới hạn. Không tối ưu C3-T bằng cách nhìn FMB/MFNet Test.**
