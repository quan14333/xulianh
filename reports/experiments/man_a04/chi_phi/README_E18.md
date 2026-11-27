# E18 – Đo chi phí: SAGE có SAM / bỏ SAM / Ghép TB / (phương pháp nhóm)

**Yêu cầu của thầy:** so mô hình có SAM, không SAM, của nhóm; chỉ ra SAM nặng ở bước nào; có đáng không;
tự chạy lại baseline và bấm giờ; chỉ ra điểm yếu phải có bằng chứng.

**Mục tiêu của nhóm:** phương pháp nhóm rẻ hơn SAGE (không cần SAM), mIoU cao hơn cả SAGE lẫn TB,
và chỉ số ghép không thua SAGE. Tất cả nằm trong cùng một bảng.

## Đo gì

| Phần | Nội dung |
|---|---|
| `tinh` | Số tham số; GFLOPs (fvcore, 1 nhân-cộng = 1 FLOP); dung lượng checkpoint |
| `sam_mask` | SAM tạo mask đúng tham số code gốc (128×128 điểm, crop_n_layers=1) trên IR và VIS → ước tính 1.220 cặp; số mask; dung lượng cache |
| `train` | Giây/bước + VRAM đỉnh: bước Student của SAGE, bước Teacher (có hoặc chưa có X-Decoder), bước bỏ SAM (E17) → ước tính 100 epoch (153 bước Student + 152 bước Teacher mỗi epoch) |
| `chay` | ms/ảnh lúc chạy, 600×800, batch 1: Student SAGE, Ghép TB |

Chỉ đo phần tính trên GPU. Không tính đọc dữ liệu và ghi ảnh/log, vì phần này giống nhau cho mọi phương pháp.
B3 (train SegFormer chấm mIoU) là thước đo chung, không tính vào chi phí.

**Số chính thức: đo trên Kaggle T4** (`../_kaggle_nb_e18`, code `../_kaggle_e18_code`). Máy nhà chỉ để thử script.

## Kết quả thử trên RTX 3050 6 GB (KHÔNG đưa vào bảng, chỉ để thử script)

`ket_qua/thu_RTX3050.json`, ngày 17/11/2026.

| | Số đo |
|---|---|
| Tham số | Student 135.617; Teacher 2,79M; SAM ViT-B 93,7M (encoder 89,7M); encoder SAM 2 tầng dùng trong L_cs 17,9M |
| GFLOPs | Student 600×800: 31,6; Student 768×1024: **51,7** (bài ghi 52,06 trên M³FD → cách đếm khớp bài); Teacher crop 192×256: 112,8; SAM encoder 1024²: 486,4 |
| L_cs | Mỗi lần tính loss gọi encoder SAM **25 lần** |
| SAM tạo mask | ~339 s/cặp (2 cặp) → **~115 giờ cho 1.220 cặp**; 190–530 mask/ảnh nhưng chỉ lưu 20, dùng 4; cache ~22 GB |
| Bước train | SAGE bước Student 1,60 s (4,8 GB); bước Teacher **hết VRAM trên 6 GB** (chưa cả X-Decoder); bỏ SAM 0,073 s (1,0 GB) → 0,31 giờ cho 100 epoch, khớp E17 thật (~22 phút) |
| Chạy | Student 44,6 ms/ảnh; Ghép TB 0,1 ms/ảnh |

## Kết quả chính thức trên Kaggle T4 (chốt 18/11/2026)

`ket_qua/T4.json` = kernel `thanhmn123/sage-e18-chi-phi-t4` **v3** + phần SAM cấu hình gốc lấy từ lần 1
(`ket_qua/T4_lan1.json`, cùng T4). Tesla T4 14,6 GB, torch 2.11, Python 3.13, detectron2 0.6.
Mỗi loại bước train đo trong một tiến trình riêng; 20 bước (10 bước cho Teacher có X-Decoder) sau 3 bước khởi động.

### Bảng chi phí (ba dòng baseline; dòng phương pháp nhóm thêm sau)

| | SAGE có SAM | SAGE bỏ SAM (E17) | Ghép TB |
|---|---|---|---|
| Tham số lúc chạy | 135.617 (Student) | 135.617 | 0 |
| Mô hình cần lúc train | Student + Teacher 2,79M + SAM ViT-B 93,7M + X-Decoder Focal-L 341,6M | Student | – |
| GFLOPs chạy (600×800) | 31,6 | 31,6 | ~0 |
| ① SAM tạo mask, 1.220 cặp | **99,2 giờ** (292,6 s/cặp, cấu hình gốc); 3,8 giờ nếu dùng cấu hình mặc định của SAM | 0 | 0 |
| ② Train 100 epoch | **26,3 giờ** = bước Student 7,6 h + bước Teacher 18,6 h | **0,50 giờ** | 0 |
| Tổng ① + ② | **≈ 125 giờ** (≈ 30 giờ nếu SAM cấu hình mặc định) | **0,5 giờ** | 0 |
| VRAM đỉnh lúc train | 11,9 GB ở batch 2; **batch 4 gốc không vừa T4 16 GB** | 1,1 GB | – |
| Dung lượng thêm lúc train | SAM 358 MB + X-Decoder 1.303 MB + cache mask ~22 GB | 0 | 0 |
| Chạy (ms/ảnh, FP32, batch 1) | 54,3 ± 0,6 (VRAM 344 MB) | như SAGE (cùng kiến trúc) | 0,05 |
| Dung lượng mô hình chạy | 0,6 MB | 0,6 MB | 0 |

### Chi tiết từng bước train (giây/bước, VRAM đỉnh)

| Bước | batch 4 | batch 2 | batch 1 |
|---|---|---|---|
| Bỏ SAM | 0,118 s – 1,1 GB | | |
| SAGE bước Student (Teacher forward + L_cs gọi encoder SAM 25 lần) | 1,798 s – 4,8 GB | | |
| SAGE bước Teacher, đủ X-Decoder (L_seg) | HẾT VRAM (suy từ VRAM batch 2) | **2,207 s – 11,9 GB** | 1,226 s – 6,7 GB |
| SAGE bước Teacher, chưa X-Decoder (chỉ tham khảo) | HẾT VRAM | HẾT VRAM* | 0,741 s – 13,8 GB* |

\* Đo ngay sau lần hết VRAM trong cùng tiến trình nên VRAM bị lẫn (bản có X-Decoder chạy ở tiến trình sạch chỉ cần 11,9 GB ở batch 2). Không dùng dòng này.

**Cách ra 26,3 giờ:** mỗi epoch 153 bước Student + 152 bước Teacher (code gốc xen kẽ batch chẵn/lẻ, batch 4).
Bước Teacher batch 4 không chạy được trên T4, nên tính bằng 2 bước batch 2 (2 × 2,207 s = 4,41 s). Đây đúng là
cách duy nhất để train SAGE trên T4, nên con số là thời gian thật trên máy này, không phải ước đoán lạc quan.
100 × (153 × 1,798 + 152 × 4,41) / 3600 = 26,3 giờ.

### SAM nặng ở bước nào? (trả lời câu hỏi của thầy)

1. **Tạo mask: 99,2 / 125 giờ ≈ 79 % tổng chi phí.** Code gốc rải 128×128 điểm + cắt 4 mảnh (khoảng 32.768 lần hỏi SAM mỗi ảnh),
   tìm 100–500 mask/ảnh nhưng chỉ lưu 20 và dùng 4. Đổi về cấu hình mặc định của SAM vẫn tốn 3,8 giờ.
2. **Bước Teacher có X-Decoder: 18,6 giờ (≈ 15 %)**, và cần > 16 GB VRAM ở batch gốc.
3. **L_cs làm bước Student chậm 15 lần** so với bỏ SAM (1,80 s so với 0,118 s), vì mỗi lần tính loss gọi encoder SAM 25 lần.
4. **Lúc chạy SAM không tốn gì thêm**: Student SAGE và Student bỏ SAM là cùng một mạng (135.617 tham số, 31,6 GFLOPs, ~54 ms/ảnh).

So sánh: SAGE tốn khoảng **125 giờ T4**, bỏ SAM tốn **0,5 giờ** (gấp ~250 lần; gấp ~60 lần nếu SAM ở cấu hình rẻ nhất).
"Có đáng không" phải đặt cạnh mIoU B3 và chỉ số ghép của từng dòng (ghép vào bảng tổng khi có đủ số).

### Kiểm số và giới hạn

- Student 768×1024 = 51,74 GFLOPs, bài ghi 52,06 → cách đếm khớp bài.
- Bỏ SAM 0,50 giờ khớp cỡ E17 thật (~22 phút trên máy khác).
- Giữa hai lần chạy trên T4: bước Student 1,61 → 1,80 s, chạy 52,1 → 54,3 ms (dao động ~4–12 % giữa các phiên Kaggle). Bảng dùng số v3.
- SAM tạo mask đo trên 5 cặp ảnh rồi nhân lên 1.220 cặp: 281–307 s/cặp (sd 9,8 s) dù số mask 106–512 → **KTC 95 %: 95–103 giờ**. Thời gian do lưới điểm cố định (~32.768 lần hỏi/ảnh) quyết định, gần như không phụ thuộc ảnh, nên 5 cặp là đủ.
- Chỉ đo phần tính trên GPU; không tính đọc dữ liệu, ghi ảnh/log (giống nhau cho mọi phương pháp).
- Mask SAM trong bước train là mask giả (vùng chữ nhật): lượng tính không phụ thuộc nội dung mask.

## Thêm phương án nhóm a04 (đo 27/11/2026, Kaggle T4)

`ket_qua/T4_a04.json` = kernel `thanhmn123/sage-e18-a04-t4` v1 (dataset `sage-e18-chi-phi` bản 27/11: `do_chi_phi.py` thêm a04 + bước tbsua,
`tbsua_992.pt` md5 fb0f5e8b…). a04 = TB + 0,4·(tbsua_992 − TB), tbsua_992 = clamp(TB + 0,5·Student(Y, IR)).
Kiểm script: Y a04 tính trong script khớp ảnh a04 đã chấm (lệch TB 0,27 mức xám, do làm tròn PNG + đổi màu).

**Kiểm máy (đo lại trong cùng phiên):** Student SAGE 52,1 ms/ảnh (18/11: 54,3); bước bỏ SAM 0,109 s (18/11: 0,118)
→ máy hôm nay nhanh hơn ~4–8 %, cùng mức → dùng tiếp số E18 cho SAGE có SAM (tạo mask, bước Student/Teacher).

### Bảng chi phí đủ 4 cột

| | SAGE có SAM | SAGE bỏ SAM (E17) | Ghép TB | **a04 (nhóm)** |
|---|---|---|---|---|
| Tham số lúc chạy | 135.617 | 135.617 | 0 | **135.617** (Student tbsua_992) |
| GFLOPs chạy (600×800) | 31,6 | 31,6 | ~0 | **31,6** (+ phép trộn, không đáng kể) |
| SAM tạo mask | 99,2 giờ | 0 | 0 | **0** |
| Train 100 epoch (1.220 ảnh, ngoại suy) | 26,3 giờ | 0,46 giờ* | 0 | **0,47 giờ*** |
| Train thật của bản dùng | – | – | – | **0,38 giờ** ước tính T4 (992 ảnh); thật 13,2 phút trên RTX 3050 |
| VRAM đỉnh lúc train | 11,9 GB (batch 2) | 1,1 GB | – | **1,1 GB** |
| Chạy (ms/ảnh, FP32, batch 1) | 52,1 ± 0,6* (54,3 ngày 18/11) | như SAGE | 0,05 | **53,9 ± 0,5** |
| VRAM lúc chạy | 336 MB | 336 MB | 8 MB | **337 MB** |
| Dung lượng mô hình | 0,6 MB | 0,6 MB | 0 | **0,6 MB** |
| Cần nhãn? | có (L_seg Teacher) | không | không | **không** |

\* đo 27/11 cùng phiên với a04. Bước tbsua 0,1095 s vs bỏ SAM 0,1089 s: loss khác nhưng lượng tính như nhau.

**Đọc:** lúc chạy a04 ngang SAGE (+1,8 ms/ảnh ≈ +3,5 %, phần trộn với TB). Lúc train a04 rẻ hơn SAGE khoảng **270 lần**
(0,47 so với ≈125 giờ gồm tạo mask), không cần SAM, X-Decoder hay nhãn.
