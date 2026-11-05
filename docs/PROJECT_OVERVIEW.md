# Tổng quan dự án

## Đề tài

**SAGE Research – Infrared-Visible Image Fusion** nghiên cứu và đánh giá việc dùng semantic
prior cho hợp nhất ảnh hồng ngoại–khả kiến, dựa trên paper CVPR 2025 *Every SAM Drop Counts:
Embracing Semantic Priors for Multi-Modality Image Fusion and Beyond* và source chính thức
<https://github.com/RollingPlain/SAGE_IVIF>.

## Phạm vi

1. Tái lập inference của pretrained SAGE Student trên FMB với preprocessing đã xác minh.
2. Xây evaluation pipeline EN, SD, SCD và MS-SSIM có kiểm chứng công thức.
3. Chọn tối đa 1–2 research direction dựa trên evidence từ paper hoặc dev experiment.
4. Prototype cải tiến nhẹ/robustness phù hợp giới hạn Kaggle GPU.
5. Báo cáo minh bạch paper result, reproduced result và illustrative result.

Không mặc định train lại Teacher/SAM. Chạy pretrained Student không đồng nghĩa full training,
và không được tuyên bố tái lập 61.2% mIoU nếu thiếu downstream evaluator/protocol tương ứng.

## Trạng thái khởi tạo

Repository chỉ có scaffold quản lý nghiên cứu. Chưa import notebook, model, dataset, checkpoint
hay kết quả thí nghiệm thật. Deadline đồ án: **17/12/2026**.
