# Checklist bàn giao LocBe — cập nhật 26/11/2026 (cho hạn 27/11)

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
| Quyền thư mục checkpoint FMB | **SECURITY ACTION REQUIRED** | 26/11 kiểm tra vẫn `Anyone: Writer`; đổi Viewer/Restricted trước chia sẻ |
| Train/Dev/Test manifest / provenance FMB | **PARTIAL** | ID/hash lưu; dataset gốc & SAGE Student binary không có trọn trong snapshot |
| MFNet cross-dataset split audit | **DONE** | 784 Train original +392 Val +393 Test, `_flip` excluded; no base-ID overlap |
| MFNet fusion SAGE/TB/C3-T, metrics, 6 panel | **DONE** | `evidence/mfnet_seed0/` — 4707 manifest rows, chưa lưu tất cả PNG binary |
| MFNet SegFormer-B3 9 lớp SAGE vs C3-T | **DONE SEED0 ONLY** | 100 epochs; Test 53.4829 vs 54.0895 mIoU9; Val 57.5254 vs 57.2138 |
| MFNet SAGE + C3-T `last.pt` | **PENDING DRIVE UPLOAD** | SHA256 đã ghi trong `MFNET_RESULTS_26_11.md`; binary không ở ZIP report |
| MFNet full 3 seed, TB B3 baseline | **OPTIONAL / NOT DONE** | Không suy diễn significant/robust gain từ seed0 |
| SemanticRT camera-held-out | **NOT DONE** | Là dataset khác MFNet, cần thí nghiệm riêng nếu nhóm yêu cầu |
| Git branch LocBe | **DONE THROUGH `bb53a37`** | `bb53a374d3c7cf2f94918c715b50ebfdf1065a24`, bao gồm MFNet evidence |
| Final handoff note/checksum package | **PREPARED; AWAITING FINAL COMMIT / DRIVE UPLOAD** | `FINAL_HANDOFF_27_11.md`, snapshot ZIP tạo riêng; script chỉ stage, không auto push |
| Draft PR to main | **PENDING** | Mở PR `LocBe → main` sau khi sửa sharing, yêu cầu Quân review |

**Đọc `FINAL_HANDOFF_27_11.md` để biết toàn bộ giới hạn. Không tối ưu C3-T bằng cách nhìn FMB/MFNet Test.**
