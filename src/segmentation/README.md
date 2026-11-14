# `src/segmentation/` – SegFormer-B3 trên FMB (baseline B3)

Chấm ảnh ghép bằng phân đoạn như Bảng 1 của bài SAGE: mỗi loại ảnh đầu vào train **một** SegFormer-B3 riêng
(cùng cấu hình), rồi đo mIoU trên FMB test. Kết quả đã khóa và protocol đầy đủ:
[`reports/experiments/b3_fmb_seg/README.md`](../../reports/experiments/b3_fmb_seg/README.md).

Owner: Mân.

## File

| File | Làm gì |
|---|---|
| `chung.py` | Đường dẫn, đọc nhãn, ma trận nhầm lẫn, mIoU theo nhiều quy ước (`cac_miou`) |
| `xuat_anh.py` | Bước 1: tạo ảnh đầu vào cho từng biến thể: `vis`, `ir`, `tb` = (Y_vis + IR)/2, `sage` (Student pretrained), `clahe` |
| `train_seg.py` | Bước 2: train SegFormer (`--gop_dev`: 1.220 ảnh, lấy `last.pt`; `--lop0`: 15 lớp) |
| `eval_seg.py` | Bước 3: chấm test → `test_nham_last.npz`, `test_ket_qua_last.json`, `test_iou_lop_last.csv`, `pred_last/` |
| `chay_song_song.py` | Chạy bước 2 + 3 cho nhiều biến thể/seed, mỗi GPU một việc (notebook Kaggle gọi file này) |
| `tong_hop.py` | Gộp mọi run → `tong_hop.csv`, bảng so với bài |
| `so_sanh_b3.py` | Chênh lệch giữa các biến thể + khoảng tin cậy 95% (bootstrap theo khối), từng seed và gộp seed |
| `ca_thang_thua.py` | Ảnh SAGE thắng/thua VIS nhiều nhất, lớp quyết định, hình minh họa |
| `thong_ke.py`, `thong_tin_anh_test.csv` | Bootstrap theo khối; khối id liền nhau, độ sáng VIS, cờ rò của 280 ảnh test |

## Đường dẫn (biến môi trường)

| Biến | Mặc định trong repo |
|---|---|
| `SAGE_FMB` | `data/raw/FMB` (tự dò thư mục `train/Visible`, `test/Visible`) |
| `SAGE_MAN` | `data/manifests` |
| `SAGE_STUDENT` | `checkpoints/pretrained/sage_student` – cần `model.py` (= `model_sub/model.py` của SAGE_IVIF) và `checkpoints.pt` (md5 `241396a51a3c5e85e360a409d9e8fe4d`) |
| `SAGE_ANH` | `outputs/b3_anh` |
| `SAGE_RUNS` | `runs/b3_fmb_seg` |

## Chạy

Cần GPU cho bước 1 (`sage`) và bước 2. Thêm thư viện ngoài `requirements.txt`: `torch`, `transformers`, `opencv-python`, `pandas`, `matplotlib`.

```powershell
python src/segmentation/xuat_anh.py --bien_the tb
python src/segmentation/train_seg.py --bien_the tb --model b3 --seed 0 --gop_dev --lop0 --epochs 100 --batch 8
python src/segmentation/eval_seg.py --run runs/b3_fmb_seg/tb_b3_s0_1220_15lop
python src/segmentation/tong_hop.py
python src/segmentation/so_sanh_b3.py
```

Trên Kaggle: [`notebooks/man/sage_b3_kaggle.ipynb`](../../notebooks/man/sage_b3_kaggle.ipynb) (T4 x2, ~2,6 giờ/run).

## Thêm phương pháp mới vào cùng thiết lập

Thêm một nhánh `--bien_the <ten>` trong `xuat_anh.py` (tạo ảnh ghép từ VIS + IR), rồi train/chấm như trên với
seed 0, 1, 2. So với `sage` và `tb` bằng `so_sanh_b3.py` (sửa danh sách `BT`, `CAP`).
