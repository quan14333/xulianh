# Bàn giao Lộc lớn — SAGE Student trên M³FD Fusion

Phạm vi theo phân công ngày 14/11/2026: **300 ảnh fused RGB PNG và EN/SD/SCD/MS-SSIM**.
Không cần huấn luyện SegFormer, SAM hoặc nhãn segmentation.

## Nguồn và tái lập

- Upstream: <https://github.com/RollingPlain/SAGE_IVIF>, MIT, commit
  `377fdbef6b8b6ccbd1139358ad66bb41ea084dc1`.
- Student: `model_sub/model.py`, load state dict `strict=True`, 135617 tham số.
- Checkpoint SHA-256: `ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836`.
  Khớp checkpoint trong `reports/experiments/sage_fmb_baseline/run_config.json`.
- Dataset: `data/raw/M3FD_Fusion/Ir` và `Vis`, đúng 300 ID; không dùng
  `M3FD_Detection-001`. Ghép theo stem, từ chối thiếu/trùng ID hoặc khác kích thước.
- Manifest ghi SHA-256 từng ảnh; `inference.csv` ghi SHA-256 ảnh đầu ra.
- Chạy `Network(VIS_Y, IR)` nguyên độ phân giải, batch 1, seed 2, eval/inference mode.
- OpenCV đọc BGR, IR chuyển GRAY, VIS chuyển YCrCb uint8, tensor float32 /255.
- Giữ thứ tự Cr/Cb và ma trận `utils.YCrCb2RGB` upstream (tên biến cb/cr trong
  upstream đảo tên nhưng dữ liệu thực tế vẫn là Y,Cr,Cb). Clamp, nhân 255, cộng 0.5,
  ép uint8, lưu RGB PNG, tương đương phép lượng tử `torchvision.utils.save_image`.
- Adapter dùng import thực tế `model_sub/model.py` vì upstream `test.py` trỏ tới
  `model_sub/model_sub.py` không tồn tại. Ma trận màu đặt trên device được chọn
  thay cho `.cuda()` cố định; không sửa kiến trúc, checkpoint hoặc upstream.

## Protocol `loclon_m3fd_y_reflect_v2`

Chấm **ảnh PNG đã lưu**, không chấm tensor trước clamp. Cả ba ảnh dùng
`Y=0.299R+0.587G+0.114B`, float64, miền 0–255. Không resize, không min–max
normalization, không nhân hệ số đồ thị khi báo metric chính.

| Metric | Quy ước |
|---|---|
| EN | `-sum(p log2 p)`, 256 bin cố định `[0,256)`, lượng tử theo floor; ảnh hằng = 0 |
| SD | Population standard deviation, `ddof=0` |
| SCD | `corr(F−VIS,IR)+corr(F−IR,VIS)`, trừ float64; correlation zero variance = 0 |
| MS-SSIM | Gaussian 11×11 sigma 1.5, reflect padding; K1=.01, K2=.03, L=255; 5 scale, weights .0448/.2856/.3001/.2363/.1333 |

MS-SSIM lấy **mean của bản đồ CS** ở từng scale trước scale cuối và **mean của
bản đồ SSIM** ở scale cuối, clamp mean âm về 0 trước lũy thừa. Downsample bằng
trung bình block 2×2, bỏ hàng/cột lẻ cuối. Mọi ảnh M³FD hiện có đều đủ 5 scales.
Metric chính là `(MS-SSIM(F,IR)+MS-SSIM(F,VIS))/2`; đồng thời xuất hai thành phần
và tổng, để tránh nhầm mean với sum. Với ảnh synthetic nhỏ, thư viện có fallback
giảm số scales và chuẩn hóa trọng số; không áp dụng fallback cho bộ 300 này.

Đây là **protocol của nhóm**. Chưa xác nhận tương đương script metric của paper
SAGE; không dùng số này để khẳng định tái lập chính xác số công bố.
Tham khảo công thức: <https://www.cns.nyu.edu/pub/eero/wang03b.pdf>.

Phiên bản cũ nhân trung bình của các thành phần thay vì trung bình tích bản đồ;
v2 sửa SSIM/MS-SSIM và khóa range histogram EN. Các bảng verification/parity
trước 14/11 là evidence của phiên bản cũ, **không xác minh v2**. Test mới dùng
Gaussian 2D trực tiếp độc lập với convolution separable trong production, gồm
trường hợp độ sáng thay đổi theo vùng và phép tổng hợp nhiều scales.

## Cách chạy

```bash
git clone https://github.com/RollingPlain/SAGE_IVIF.git external/SAGE_IVIF
git -C external/SAGE_IVIF checkout 377fdbef6b8b6ccbd1139358ad66bb41ea084dc1
# Chọn PyTorch CPU/CUDA phù hợp trước, sau đó:
.venv/bin/python -m pip install -r requirements-inference.txt
.venv/bin/python scripts/run_m3fd_baseline.py --audit-only
.venv/bin/python scripts/run_m3fd_baseline.py --device cpu --threads 4
# Hoặc trên máy có GPU: --device cuda
```

Runner tạo thư mục run mới, từ chối ghi đè, lưu config/environment/manifest/log,
300 ảnh `fused/`, `inference.csv`, `metrics/per_image.csv`, `metrics/summary.json`
và preview 6 ID cách đều theo danh sách đã sắp xếp. Chỉ ghi `status=complete`
khi đủ 300 ảnh đúng ID và tất cả metric hữu hạn. Run lỗi hoặc bị ngắt vẫn giữ
tiến độ và đánh dấu `failed_or_interrupted`, không coi là bàn giao hoàn tất.

Summary báo mean/std/median/valid count; std giữa các ảnh cũng dùng ddof=0.
Không bỏ ảnh lỗi để báo kết quả đẹp. Artifact ảnh và checkpoint được ignore;
bảng nhỏ và báo cáo chỉ tới run thật. Chênh lệch CPU/CUDA có thể gây khác biệt
floating point nhỏ; dùng environment và SHA-256 để nhận diện đúng run.
