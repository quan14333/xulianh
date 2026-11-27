# MeanRes – kết quả (phần của Mân)

*Phương pháp: [`docs/methods/man_a04_METHOD_SUMMARY.md`](../../../docs/methods/man_a04_METHOD_SUMMARY.md).
Code: [`src/methods/man_a04/`](../../../src/methods/man_a04/README.md). Khóa 26/11/2026 trên Dev228 → [`FROZEN_MANIFEST.json`](FROZEN_MANIFEST.json).
Giao thức chấm + câu kết luận ghi trước khi mở số Test: [`QUY_TAC_CHAM.md`](QUY_TAC_CHAM.md).*

**MeanRes** (mã trong code: `a04` / `tbsua_a04`, a = alpha 0,4): Y = TB + 0,4·(tbsua_992 − TB), TB = (Y_vis + IR)/2; tbsua_992 là Student 135.617 tham số học **không nhãn, không SAM**.

## 1. Tóm tắt

| | FMB Test (273 ảnh sạch, 5 seed) | Ý nghĩa |
|---|---|---|
| MeanRes − SAGE | **+0,82** (5/5 seed dương, CI ảnh [0,15; 1,43]) | **hơn nhất quán** |
| MeanRes − TB | −0,17 (3/5 seed dương); CI ảnh [−0,47; 0,12], CI ảnh + seed [−0,68; 0,44] | **ngang** (CI ảnh trong ±0,5); chưa khẳng định chặt khi tính cả seed |
| Chi phí train | 0,47 giờ so với ≈ 125 giờ của SAGE | không SAM, X-Decoder, nhãn |
| Chi phí chạy | 53,9 so với 52,1 ms/ảnh (T4) | ngang SAGE (+3,5 %) |

Mục tiêu "hơn nhất quán cả SAGE lẫn TB" **chưa đạt**: MeanRes hơn SAGE, ngang TB. Cả hai khoảng tin cậy MeanRes − TB đều chứa 0,
nên **không** kết luận MeanRes thua TB. Biên −0,5 trong `QUY_TAC_CHAM.md` chưa được nhóm chốt → báo cả hai CI (đính chính 27/11 trong file đó).

## 2. FMB Test – mIoU 13 lớp (`test_fmb/`)

Giao thức: SegFormer-B3 train trên ảnh ghép của 1.220 ảnh train, 100 epoch, `last.pt`, 15 lớp; một ma trận nhầm lẫn gộp
toàn tập, bỏ pixel nhãn 0, trung bình 13 lớp có mặt (bicycle không có trong test). Số chính trên **273 ảnh** (bỏ 7 ảnh rò), kèm 280.

| | s0 | s1 | s2 | s3 | s4 | Trung bình ± SD | 280 ảnh |
|---|---|---|---|---|---|---|---|
| **MeanRes** | 63,65 | 64,26 | 65,22 | 63,04 | 62,46 | **63,73 ± 1,07** | 63,47 |
| TB | 64,27 | 63,91 | 65,09 | 62,94 | 63,29 | 63,90 ± 0,84 | 63,68 |
| SAGE | 62,62 | 63,07 | 64,42 | 62,64 | 61,76 | 62,90 ± 0,97 | 62,75 |
| nosam (3 seed) | 62,89 | 64,07 | 64,46 | – | – | 63,81 ± 0,81 | 63,51 |

So theo cặp seed (`so_sanh_cap.py`, bootstrap khối 10 ảnh, B = 2000; thang 3 mức trong `QUY_TAC_CHAM.md`):

| Cặp | Hiệu TB | Seed dương | CI ảnh | CI ảnh + seed | Kết luận |
|---|---|---|---|---|---|
| MeanRes − SAGE | +0,82 | 5/5 | [0,15; 1,43] | [−0,13; 1,52] | hơn nhất quán |
| MeanRes − TB | −0,17 | 3/5 | [−0,47; 0,12] | [−0,68; 0,44] | chưa phân biệt (ngang theo CI ảnh, ±0,5) |
| TB − SAGE | +1,00 | 5/5 | [0,20; 1,76] | [−0,17; 1,84] | hơn nhất quán |
| MeanRes − nosam (s0–2) | +0,57 | 3/3 | [−0,07; 0,92] | [−0,26; 1,22] | chưa phân biệt |

Trên 280 ảnh: MeanRes − SAGE +0,72 (CI ảnh [−0,05; 1,33], chưa phân biệt), MeanRes − TB −0,21.

### IoU theo lớp (trung bình 5 seed, 273 ảnh) – `test_fmb/results_per_class.csv`

Cột `nosam` trong CSV là trung bình **3 seed** (s0–2); cả 3 seed tính lại từ `test_nham_last.npz`. Run nosam s1, s2 không kèm
`test_iou_lop_last.csv`; IoU từng lớp của chúng lấy từ ma trận nhầm lẫn trong `.npz` (cùng cách tính với mọi run).

| Lớp | MeanRes | TB | SAGE | MeanRes − TB | MeanRes − SAGE |
|---|---|---|---|---|---|
| road | 88,3 | 88,0 | 87,7 | +0,3 | +0,6 |
| sidewalk | 44,7 | 43,8 | 43,3 | **+0,9** | **+1,5** |
| building | 84,7 | 84,8 | 84,3 | −0,1 | +0,4 |
| traffic_light | 38,2 | 39,0 | 41,0 | **−0,8** | **−2,7** |
| traffic_sign | 79,0 | 79,1 | 78,3 | −0,1 | +0,7 |
| vegetation | 87,8 | 87,9 | 87,3 | −0,1 | +0,5 |
| sky | 95,5 | 95,7 | 95,4 | −0,2 | +0,1 |
| person | 69,7 | 69,6 | 70,2 | +0,1 | −0,5 |
| car | 79,3 | 78,6 | 78,8 | **+0,7** | +0,4 |
| truck | 30,6 | 31,4 | 31,0 | **−0,8** | −0,4 |
| bus | 37,5 | 39,4 | 31,0 | **−1,9** | **+6,5** |
| motorcycle | 42,1 | 42,1 | 38,6 | −0,1 | **+3,5** |
| pole | 51,0 | 51,3 | 50,9 | −0,2 | +0,2 |

- **Tăng nhiều nhất so với SAGE:** bus (+6,5), motorcycle (+3,5), sidewalk (+1,5). Ở bus và motorcycle, TB cũng hơn SAGE
  tương tự (+8,4, +3,5) → giả thuyết: lợi ích chủ yếu đến từ việc **giữ cường độ IR/VIS kiểu TB**, không phải từ phần sửa.
  Riêng sidewalk, MeanRes hơn cả TB (+0,9).
- **Giảm so với SAGE:** traffic_light (−2,7), person (−0,5), truck (−0,4). Đèn và người là vùng nhỏ, tương phản cao;
  Giả thuyết: SAGE (có L_cs theo vùng SAM) làm nổi các vùng này hơn. Trên ảnh SemanticRT train, tương phản vùng người của SAGE
  gấp gần 2 lần MeanRes (79,8 so với 42,9).
- **So với TB:** MeanRes hơn ở sidewalk/car/road, thua ở bus/truck/đèn; mọi chênh lệch theo lớp đều < 2 điểm.
  Giả thuyết (chưa kiểm): phần sửa làm rõ cạnh/mép đường, nhưng đẩy cường độ lệch khỏi TB ở vật lớn đồng màu (bus, truck).

## 3. Dev228 – căn cứ khóa (`dev228/`)

Split **scene_v1** (nhóm chốt 16–18/11): 992 Train / 228 Dev, tách từ 1.220 ảnh train chính thức **theo cảnh**
(Dev 175 cũ + 53 ảnh train cùng cảnh chuyển sang; 10 khối id liền). File: `dev228/split_scene_v1/train.csv`, `dev.csv`
(md5 `e3979a3d…`, `7969b4e3…`), lý do từng nhóm ảnh: `dev228/split_scene_v1/scene_decision.md`.
**Lưu ý có 2 split:** Dev228 (992/228) dùng để train `tbsua_992` và chọn phương án; Test cuối dùng manifest 1.045/175/280 trong
`goi_test/manifests/` với `--gop_dev` (SegFormer train trên đủ 1.220 ảnh), nên hai split cho cùng tập train 1.220.

3 seed, 228 ảnh Dev, mIoU 13 lớp: MeanRes **60,77**, TB 60,15, SAGE 59,41.
MeanRes − TB +0,63 (CI ảnh [−0,04; 1,26]), MeanRes − SAGE +1,36 (CI ảnh [0,06; 3,09]). `a04_cvis` (MeanRes thêm cổng VIS) rớt luật
đặt trước (59,41 < 60,69) → giữ a04. **Ghi nhận:** luật phụ 8 lớp phổ biến đặt trước rớt (77,48 < 77,62).

Chỉ số ghép Dev228 (`dev228/rao_a04_cvis.txt`):

| | SCD | MS-SSIM* | Q_ABF | EN | SD |
|---|---|---|---|---|---|
| SAGE | 1,698 | 1,569 | 0,612 | 6,65 | 36,7 |
| **MeanRes** | 1,477 (87,0 %) | 1,629 (103,8 %) | 0,648 (105,9 %) | 6,30 | 25,4 |

\* MS-SSIM là **tổng** MS-SSIM với hai ảnh nguồn (như bài SAGE), nên có thể > 1. Rào đặt trước: MS-SSIM ≥ 99 %, Q_ABF ≥ 98 % SAGE → qua.
SCD 87 % SAGE: MeanRes giữ tương quan với từng nguồn kém hơn SAGE (chỉ báo kèm, không loại).

## 4. SemanticRT – kiểm tổng quát (`semanticrt/`, đang chạy)

Chia lại theo **địa điểm** (split gốc có 347 ảnh test trùng train): 1.000 train / 300 test (75 % ảnh đêm), xem
`semanticrt/chia_lai_README.md`. File split + script chia: `semanticrt/split/` (`train.csv`, `test.csv` là bản chia gốc;
`kaggle_train.csv`, `kaggle_test.csv` là bản rút gọn cột đưa lên Kaggle, cùng id). `thong_tin.csv` (1,7 MB) không kèm,
tạo lại bằng `b1_thong_tin.py <thư mục SemanticRT>`.
Mô hình ghép giữ nguyên (chỉ học trên FMB). mIoU 13 lớp gồm background.
Tên run có hậu `_1220` chỉ là quy ước tên chung với FMB; SegFormer ở đây train trên **1.000** ảnh SemanticRT.

| | MeanRes | SAGE | TB | VIS |
|---|---|---|---|---|
| seed 0 | **48,31** | 47,84 | 47,79 | 42,39 |

Mới seed 0; luật cần seed 0–2. Tình trạng 27/11: seed 1 (MeanRes + SAGE) đang chạy (notebook `sage-srt-a04-s1`);
TB seed 1 và cả seed 2 chạy sau khi quota reset 28/11. Luật đọc đặt trước nằm ở mục SemanticRT của `QUY_TAC_CHAM.md`;
tên notebook `sage-srt-b3-s{0,1,2}` ghi trong đó là tên cũ (đã xóa, thay bằng `sage-srt-a04-*`), luật giữ nguyên. Số tuyệt đối thấp vì train 1.000 ảnh + test là địa điểm chưa thấy;
**không so được với bài SemanticRT** (khác split, khác số ảnh train). Chỉ số ghép trên 300 ảnh train
(`semanticrt/rao_anh_train300.txt`): MeanRes MS-SSIM 106,7 %, Q_ABF 116,3 %, SCD 93,4 % SAGE.

## 5. Chi phí (`chi_phi/`, Kaggle T4)

| | SAGE có SAM | SAGE bỏ SAM | Ghép TB | **MeanRes** |
|---|---|---|---|---|
| Tham số lúc chạy | 135.617 | 135.617 | 0 | 135.617 |
| SAM tạo mask (1.220 cặp) | 99,2 giờ | 0 | 0 | 0 |
| Train 100 epoch | 26,3 giờ | 0,46 giờ | 0 | 0,47 giờ |
| VRAM lúc train | 11,9 GB | 1,1 GB | – | 1,1 GB |
| Chạy (ms/ảnh, FP32, 600×800) | 52,1 | như SAGE | 0,05 | 53,9 |
| VRAM lúc chạy | 336 MB | 336 MB | 8 MB | 337 MB |
| Cần nhãn phân đoạn khi train | có (L_seg) | không | không | không |

SAGE có SAM: số train/tạo mask đo ngày 18/11 (`chi_phi/T4.json`); số chạy và các cột còn lại đo 27/11 trong cùng phiên
(`chi_phi/T4_a04.json`). Bước bỏ SAM đo lại 0,109 s so với 0,118 s ngày 18/11 → máy cùng mức.

## 6. Không làm và lý do

- **TTA:** MeanRes là phương pháp ghép ảnh; TTA thuộc bộ phân đoạn, áp như nhau cho mọi ảnh ghép → không phải baseline so sánh.
  Mọi bản dùng cùng SegFormer-B3, không TTA.
- **MFNet:** chưa làm vì giới hạn GPU; kiểm tổng quát dùng SemanticRT.

## 7. Tái lập

- Dựng ảnh + train/chấm SegFormer trên Kaggle: `src/methods/man_a04/goi_test/` (đúng byte gói đã chạy Test, xem SHA256 trong manifest).
- Chấm cặp seed: `python goi_test/so_sanh_cap.py --bien_the tbsua_a04 tb sage --model b3 --hau _1220_15lop` với `SAGE_RUNS` trỏ vào `test_fmb/runs`.
- Bảng CSV: `src/methods/man_a04/cham/xuat_bang_ket_qua.py`.
- Run tb/sage seed 0–2 là mốc baseline B3 (cùng cấu hình, `reports/experiments/b3_fmb_seg/`); seed 3, 4 và MeanRes chạy 26/11.
- Thư viện: `requirements.txt` (bản Kaggle đã chạy Test: Python 3.13, torch 2.11.0+cu130, Tesla T4).
- Đo tốc độ: `runtime_benchmark.json` (chép từ `chi_phi/T4_a04.json`, cùng nội dung).
- Hình ca Test: `src/methods/man_a04/hinh/hinh_ca_test.py` (mục 8).

## 8. Hình ca Test (`figures/`)

Mỗi hình: VIS | IR | nhãn GT; ảnh ghép MeanRes | TB | SAGE; mask dự đoán seed 0 của từng bản (**trắng = pixel sai**, vùng nhãn 0
làm tối vì không chấm). Mask là **đúng lượt Test đã chấm** (không train lại): MeanRes lấy `pred_last/*.png` trong output Kaggle
`sage-test-a04-nhom1`, TB/SAGE từ output lượt baseline. Kiểm: mask + nhãn → ma trận nhầm lẫn **trùng tuyệt đối** `test_nham_last.npz`
ở cả 280 ảnh, 3 bản. Ảnh ghép MeanRes dựng lại trên máy bằng đúng công thức `dung_test.py` và `tbsua_992.pt` đã khóa.

Luật chọn ca ghi trong script **trước khi** xem điểm từng ảnh: điểm = mIoU một ảnh (bỏ nhãn 0), trung bình 5 seed, 273 ảnh sạch;
cải thiện = min(MeanRes − TB, MeanRes − SAGE) lớn nhất; suy giảm = max(...) nhỏ nhất; khó = 00089 (khói, từ audit) + ảnh VIS tối nhất;
hai ca cách nhau > 10 id. Điểm mọi ảnh: `figures/diem_tung_anh.csv`.

| Hình | Ảnh | MeanRes | TB | SAGE | Ghi chú |
|---|---|---|---|---|---|
| `ca1_cai_thien_00381.png` | 00381 | 56,6 | 52,7 | 53,8 | đêm, đám đông trên bậc thang |
| `ca2_cai_thien_00503.png` | 00503 | 64,6 | 62,0 | 62,3 | phố đêm có xe; vỉa hè bên phải ít sai hơn |
| `ca3_suy_giam_00794.png` | 00794 | 63,9 | 67,4 | 70,9 | phố cao ốc chạng vạng, nhiều vật nhỏ |
| `ca4_suy_giam_00716.png` | 00716 | 49,2 | 52,1 | 52,6 | đêm sương; cả 3 sai tòa nhà bên trái, MeanRes mất hẳn mảnh building |
| `ca5_kho_khoi_00089.png` | 00089 | 90,2 | 91,1 | 87,8 | VIS bị khói, IR thấy 2 người |
| `ca6_kho_toi_00046.png` | 00046 | 71,2 | 72,2 | 73,6 | ảnh tối (00381 tối hơn nhưng đã là ca 1) |

mIoU một ảnh rất nhiễu (một lớp nhỏ đổi vài trăm pixel là lật), nên hình **chỉ để minh họa**; kết luận vẫn theo mIoU gộp toàn tập (mục 2).
Ghi chú: notebook Test xóa `last.pt` sau khi chấm (lỗi của bước dọn output, đã sửa ở script sinh notebook), nhưng mask PNG vẫn được lưu.
