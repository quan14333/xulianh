# Baseline SAGE Student trên FMB test

## Mục tiêu

Tái lập inference của pretrained SAGE Student trên FMB official test split. Đây là kết quả inference, chưa phải tái lập toàn bộ bài báo *Every SAM Drop Counts* (CVPR 2025).

## Phương pháp

Notebook đã chạy: [SAGE_Quan_FMB_TEST280_Kaggle.ipynb](../../../notebooks/quan/SAGE_Quan_FMB_TEST280_Kaggle.ipynb). Notebook lấy mã nguồn và pretrained checkpoint từ [SAGE_IVIF chính thức](https://github.com/RollingPlain/SAGE_IVIF), tại commit `377fdbef6b8b6ccbd1139358ad66bb41ea084dc1`. Mô hình nhận ảnh Visible và Infrared, xuất ảnh fused RGB. Số tham số Student: 135.617.

## Dataset

FMB official test split có 280 cặp ảnh, kích thước 800×600. Kaggle Input đã dùng: `/kaggle/input/datasets/quannh206/data-fmbtest/test`, với hai thư mục `Visible` và `Infrared`.

## Preprocessing và postprocessing

Theo `run_config.json` của lần chạy: Visible qua OpenCV BGR → YCrCb rồi `torchvision.ToTensor`; Infrared qua OpenCV BGR → GRAY; không resize. Batch size 1, seed 2, test mode không augmentation. Output dùng `utils.YCrCb2RGB` của SAGE, clamp về `[0,1]`, rồi `torchvision.utils.save_image` thành PNG. Notebook được import nguyên code cells, không sửa logic inference.

## Kết quả

Run ID: `sage_fmb_test_student_20261106T160350Z`.

| Chỉ số | Kết quả |
| --- | ---: |
| Ảnh fused | 280/280 |
| QA PASS / FAIL | 280 / 0 |
| Mean forward | 74,625 ms/ảnh |
| Mean fusion | 76,038 ms/ảnh |
| Tổng thời gian loop | 104,352 giây |
| Peak CUDA memory allocated | 510,589 MiB |
| Tham số Student | 135.617 |

Các số đo thời gian và bộ nhớ lấy từ `environment.json` do lần chạy ghi lại; validator xác minh metadata và nội dung ảnh, không đo lại tốc độ GPU.

## Tái lập

Môi trường lần chạy: Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, CUDA 13.0, Tesla T4. SHA-256 checkpoint: `ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836`. Validator đã tính lại hash từ checkpoint nằm trong ZIP; Git ref của bản source clone trong ZIP cũng khớp commit nêu trên.

Trên Kaggle, notebook cần bật **GPU** và **Internet**, gắn FMB test input ở đường dẫn trên và có quyền lấy source/checkpoint SAGE. Để lấy output gốc, mở Kaggle Notebook Versions của notebook tương ứng, chọn phiên bản chạy hoàn tất và tải **Output**, hoặc gắn notebook output làm Kaggle Input tại `/kaggle/input/notebooks/quannh206/notebookde8ef3ea09`. ZIP 425 MB, ảnh fused và checkpoint được giữ ngoài Git; xem `artifact_manifest.json` để biết archive gốc và SHA-256.

## Kiểm nghiệm

```powershell
python scripts/validate_sage_baseline.py --input "C:\Users\quan\Downloads\results (6).zip" --report reports/experiments/sage_fmb_baseline/validation_report.json
```

Lần kiểm nghiệm archive gốc: **PASS**. Script đọc ZIP trực tiếp, giải mã cả 280 PNG bằng Pillow, kiểm tra RGB 800×600, tên file, các bảng CSV, split, QA, hash checkpoint và commit trong Git ref. Chi tiết từng tiêu chí ở `validation_report.json`. Các file metadata, bảng và preview trong thư mục này được copy từ ZIP, còn report do script tạo ra.

## Giới hạn và bàn giao

- Đã hoàn thành pretrained Student inference trên 280 cặp FMB test.
- Chưa đánh giá EN, SD, SCD, MS-SSIM hoặc segmentation mIoU.
- Chưa tái lập huấn luyện Teacher–Student; không tuyên bố tái lập 61,2% mIoU của tác giả.
- Để Lộc lớn tính fusion metrics: lấy 280 PNG từ Kaggle Output, đối chiếu ID với `manifest.csv`, gắn ảnh Visible/Infrared FMB test tương ứng, chạy các metric theo `configs/evaluation/fusion_metrics.yaml`, lưu script, cấu hình và bảng kết quả. Giữ ảnh và dataset ngoài Git.
