# Lộc bé — SAGE / FMB scene_v1 Failure Analysis

Đây là gói bằng chứng **Train-only** ngày 19/11/2026, phục vụ nhiệm vụ Failure Analysis của nhóm CS406.

- Dữ liệu: FMB **scene_v1**, 992 Train / 228 Dev, đọc 992 ảnh VIS–IR–SAGE Train và (chỉ để phân tích lỗi) nhãn Train.
- 30 trường hợp ưu tiên, sáu bảng so sánh ảnh, hai giả thuyết có thể bác bỏ.
- Nguồn: bản ZIP Kaggle `20261119T043701Z.zip` SHA256 `005128d8b6472f4d9bbbec0d6095942cec189979422ffb8285d8ea677f906001`.
- `evidence/`: 10 file xuất nguyên vẹn từ ZIP, không sửa nội dung.
- `FAILURE_ANALYSIS_REVIEW_20261119.md`: diễn giải và thẩm định độc lập từ artifact gốc.
- `notebooks/`: notebook Kaggle dùng để tạo bằng chứng; cần mount đúng dữ liệu để tái chạy.

## Kết luận giới hạn

**Chưa có bằng chứng mIoU hoặc quan hệ nhân quả** giữa edge/contrast retention và segmentation; 30 ảnh không nhất thiết là 30 scene độc lập. Không dùng Dev/Test để chọn phương án. Không dùng Ground Truth khi tạo fusion ở inference.

## Bước tiếp theo

1. Quân xuất per-image/classwise IoU và prediction cho hai baseline VIS/SAGE trên Train, giữ nguyên split/recipe.
2. Thử H1 (HF-VIS residual, có và không có gate) **trên Train** với alpha được đăng ký trước; ablation zero/fixed/gated.
3. Kiểm tra cải thiện mIoU bằng đánh giá B3 hợp lệ rồi mới chốt phương án.

## Không được commit

FMB raw data, VIS/IR/SAGE full images, checkpoint `.pt`, Kaggle cache, credentials, weights.
