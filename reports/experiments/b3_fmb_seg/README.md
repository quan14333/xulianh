# Kiểm lại SAGE trên FMB bằng SegFormer-B3 (phần "B3")

*Cập nhật 14/11/2026 – **bản khóa**, đủ 3 seed (0, 1, 2). Phần chạy B3 đã xong, không chạy thêm.*

---

## 1. Tóm tắt trong 30 giây

- **Mục đích:** kiểm lại lời bài SAGE (CVPR 2025): *"ảnh ghép SAGE giúp mạng phân đoạn hiểu cảnh tốt hơn ảnh thường **+12,1 mIoU**"* trên FMB.
- **Kết quả:** nhóm đo được SAGE chỉ hơn ảnh thường **+0,8 mIoU** (trung bình 3 seed), không seed nào có ý nghĩa thống kê → **chưa tái lập được** +12,1.
- **Bước tạo ảnh SAGE của nhóm đúng:** checkpoint công bố chạy ra gần trùng ảnh mẫu của tác giả. Số của ảnh thường khớp bài, chỉ riêng SAGE thấp hơn bài. Bài không ghi ảnh SAGE ở Bảng 1 tạo từ đâu, nên **chưa xác định được nguyên nhân**.
- **Ghép đơn giản tốt hơn SAGE (sát ngưỡng):** cộng ảnh nhiệt với ảnh thường rồi chia đôi được **64,4**, cao hơn SAGE (**63,4**) ở cả 3 seed; gộp 3 seed chênh **+1,05**, có ý nghĩa nhưng sát 0.
- **SAGE giúp thật ở lớp người** (ban đêm, có khói), nhưng ảnh nhiệt đơn thuần hay ghép trung bình cũng giúp y như vậy → cái lợi đến từ **ảnh nhiệt**, chưa chắc nhờ phần SAM của SAGE.

---

## 2. Làm gì?

Bài SAGE chấm chất lượng ảnh ghép bằng cách: **lấy ảnh ghép đem train một mạng phân đoạn (SegFormer-B3), rồi đo mIoU** (mạng nhận đúng từng vùng tới đâu). mIoU càng cao = ảnh ghép càng giúp hiểu cảnh.

Nhóm làm y như vậy với **4 loại ảnh đầu vào**, mỗi loại train **một mạng riêng, cùng cấu hình**, chỉ đổi ảnh:

| Ký hiệu | Ảnh đầu vào | Vai trò |
|---|---|---|
| **VIS** | Ảnh thường (camera màu) | Mốc: không ghép |
| **IR** | Ảnh nhiệt (hồng ngoại) | Mốc: không ghép |
| **TB** | Ghép trung bình: (ảnh thường + ảnh nhiệt) / 2 | Mốc: cách ghép đơn giản nhất (bài không có) |
| **SAGE** | Ảnh ghép của SAGE (trọng số công bố) | Phương pháp đang kiểm |

> Mọi chênh lệch giữa các dòng đều đến từ **ảnh đầu vào**, vì mạng và cách train giống hệt nhau.

---

## 3. Cấu hình

Bài **không công bố code chấm** và **không ghi đủ chi tiết**. Chỗ nào bài thiếu, nhóm tự chọn theo cấu hình chuẩn và ghi rõ.

| Mục | Nhóm chọn | Bài có ghi? | Căn cứ |
|---|---|---|---|
| Mạng | SegFormer-B3 | ✅ | Theo bài |
| Khởi tạo | `mit-b3` – chỉ học trên ImageNet-1k (không phải bản Cityscapes) | ❌ | Cấu hình SegFormer gốc |
| Số lớp khi train | 15 lớp (nhãn 0 "unlabeled" là một lớp) | ❌ | Code chấm FMB của MRFS; code SegMiF (cùng nhóm tác giả SAGE) |
| Dữ liệu train | Đủ 1.220 ảnh train chính thức | ✅ | Theo bài |
| Số epoch | 100 (15.200 vòng lặp), batch 8 | ✅ epoch / ❌ batch | Theo bài / vừa GPU T4 |
| Tối ưu | AdamW, lr 6e-5 (đầu ra ×10), warmup 1.500 vòng rồi giảm poly | ❌ | Cấu hình SegFormer gốc |
| Tăng cường dữ liệu | Phóng to/thu nhỏ 0,5–2,0, cắt 512, lật ngang | ❌ | Cấu hình SegFormer gốc |
| Chọn checkpoint | Lấy lần cuối, **không chọn** | ❌ | Để không đụng tới test |
| Chấm | Nguyên ảnh 600×800; **280 ảnh** test và **273 ảnh** (bỏ 7 ảnh rò) | ❌ | |
| Số lần chạy | 3 seed (0, 1, 2) | ❌ | |
| Máy | Kaggle, GPU T4, ~2,5 giờ / mạng | | |

**Cách tính mIoU (số chính):** bỏ pixel không có nhãn, trung bình trên 13 lớp có mặt trong test (lớp xe đạp không có ảnh test nào). Có tính thêm kiểu SegMiF (15 lớp) để đối chiếu – kết luận không đổi.

---

## 4. Kết quả chính

### 4.1. mIoU (273 ảnh sạch)

| | Seed 0 | Seed 1 | Seed 2 | **TB ± SD** |
|---|---|---|---|---|
| VIS (ảnh thường) | 62,07 | 62,56 | 63,02 | **62,55 ± 0,48** |
| IR (ảnh nhiệt) | 54,66 | 56,64 | 57,93 | **56,41 ± 1,64** |
| SAGE | 62,62 | 63,07 | 64,42 | **63,37 ± 0,94** |
| **TB (ghép trung bình)** | 64,27 | 63,91 | 65,09 | **64,42 ± 0,60** |

*Kiểu SegMiF (15 lớp, 280 ảnh), trung bình 3 seed: VIS 54,0 / IR 48,9 / SAGE 54,7 / TB 55,6 – cùng thứ tự.*

### 4.2. Chênh lệch (dấu * = có ý nghĩa thống kê, khoảng tin cậy 95% không chứa 0)

| | Seed 0 | Seed 1 | Seed 2 | **Gộp 3 seed** | Đọc là |
|---|---|---|---|---|---|
| SAGE − VIS | +0,55 | +0,51 | +1,40 | +0,82 [−0,86; +3,04] | SAGE gần như **không hơn** ảnh thường |
| TB − VIS | +2,20* | +1,35 | +2,07* | +1,87 [−0,01; +4,19] | Ghép trung bình có giúp (sát ngưỡng) |
| **TB − SAGE** | **+1,65*** | +0,84 | +0,67 | **+1,05 [+0,08; +2,16]*** | Ghép trung bình **tốt hơn** SAGE, sát ngưỡng |
| SAGE − IR | +7,95* | +6,43* | +6,49* | +6,96 [+3,58; +10,46]* | SAGE hơn hẳn ảnh nhiệt đơn thuần |

Khoảng tin cậy tính bằng bootstrap theo **khối ảnh liền nhau** (ảnh FMB cắt từ video, ảnh liền số giống nhau). Cột "gộp 3 seed": mỗi lần bốc lại dùng cùng bộ ảnh cho cả 3 seed rồi lấy trung bình chênh lệch – chỉ tính dao động theo ảnh, không tính dao động giữa seed (xem 3 cột seed để thấy dao động đó).

**Theo độ sáng ảnh (gộp 3 seed, mỗi nhóm 91 ảnh):** ảnh tối TB − SAGE +3,4 [−1,6; +4,5], SAGE − VIS +2,5 [−2,9; +8,1] – lớn nhưng chưa có ý nghĩa vì ít ảnh; ảnh vừa/sáng chênh dưới ±1,1.

---

## 5. So với bài báo

**Không so thẳng 49,1 (VIS bài) với 62,3 (VIS nhóm)** – hai bên tính mIoU theo cách khác nhau (bài không nói tính trên lớp nào). Để so công bằng, dùng **7 lớp bài có in số**:

| Lớp | VIS bài | VIS nhóm | IR bài | IR nhóm | SAGE bài | SAGE nhóm | TB nhóm |
|---|---|---|---|---|---|---|---|
| Đường | 84,8 | 88,4 | 83,7 | 86,4 | 89,0 | 87,7 | 88,0 |
| Nhà | 81,0 | 82,6 | 77,6 | 82,1 | 83,9 | 84,6 | 84,8 |
| Biển báo | 64,2 | 77,3 | 55,9 | 65,7 | 71,4 | 78,3 | 79,1 |
| Cây | 84,7 | 86,3 | 80,9 | 83,2 | 86,3 | 87,4 | 87,9 |
| **Xe buýt** | 46,8 | 34,5 | 38,1 | 37,6 | **66,7** | **32,1** | 38,6 |
| Người | 61,1 | 59,0 | 57,1 | 70,9 | 64,1 | 69,9 | 69,7 |
| Xe con | 77,1 | 79,4 | 74,2 | 75,7 | 81,7 | 78,4 | 78,2 |
| **TB 7 lớp** | **71,4** | **72,5** | **66,8** | **71,7** | **77,6** | **74,0** | **75,2** |

*Số nhóm: 280 ảnh, trung bình 3 seed.*

| Mức tăng của SAGE so với VIS | Bài | Nhóm |
|---|---|---|
| Trên 7 lớp in ra | **+6,2** | **+1,6** |
| Trên mIoU chung | **+12,1** | **+0,8** |

**Đọc bảng:**
- **VIS khớp bài** (72,5 vs 71,4). IR thấp hơn VIS ở cả bài lẫn nhóm, nhưng IR nhóm cao hơn IR bài (71,7 vs 66,8).
- **SAGE lệch rõ** (74,0 vs 77,6), lệch nhiều nhất ở **xe buýt** (32,1 vs 66,7).
- VIS khớp mà SAGE lệch → nghi khác biệt nằm ở **ảnh SAGE bài đã dùng**. Nhưng bài **không ghi** ảnh SAGE ở Bảng 1 do Teacher hay Student tạo, dùng checkpoint nào, và không công bố ảnh ghép FMB để đối chiếu. IR cũng không khớp hẳn (71,7 so với 66,8), nên không loại trừ được việc cách chấm hai bên khác nhau. **Chưa xác định được nguyên nhân.**
**Ảnh SAGE của nhóm lấy từ đâu:**
- Tạo bằng **Student** (≈ 0,136M tham số) với checkpoint **duy nhất** tác giả công bố (`checkpoints/checkpoints.pt` của RollingPlain/SAGE_IVIF, md5 `241396a51a3c5e85e360a409d9e8fe4d`), giống `test.py`. Tác giả không công bố checkpoint Teacher.
- Đã kiểm (14/11): chạy checkpoint này trên 6 ảnh mẫu trong repo SAGE (`data/test`, ảnh M³FD 1024×768) ra gần trùng ảnh ghép tác giả để sẵn (`result/test`): lệch trung bình 0,2 mức xám (thang 0–255), tương quan 0,9999 → bước tạo ảnh của nhóm đúng.
- Thống kê BatchNorm lớp đầu của checkpoint (giải ngược ra độ sáng trung bình và phương sai dữ liệu lúc train) khớp **FMB train** tốt nhất → checkpoint nhiều khả năng được train trên FMB. Giới hạn: BatchNorm chỉ phản ánh giai đoạn cuối quá trình train.

- Lớp xe buýt **dao động rất mạnh giữa các seed** (TB 42,1 → 27,4 → 46,2; IR 21,6 → 39,9 → 51,2) vì hiếm: 58/280 ảnh test có xe buýt, 5 ảnh chiếm 58% pixel xe buýt. Không dùng riêng lớp này để kết luận.

---

## 6. Các bài khác chấm lại SAGE trên FMB

*(Số "theo bài", nhóm chưa tự kiểm. Tra đến 11/2026.)*

| Bài | Mạng chấm | mIoU SAGE | SAGE hơn VIS |
|---|---|---|---|
| **SAGE gốc** | SegFormer-B3 | 61,2 | **+12,1** |
| SSVIF (preprint) | SegFormer-B3 | 62,0 | +0,03 |
| AD-SRF | OverLock-B | 61,4 | +2,5 |
| P2Fusion | SegFormer-B5 | 56,5 | +1,0 |
| PCDFusion | SegFormer-B2 (khởi tạo Cityscapes) | 56,3 | +3,7 |
| CLDyN | SegFormer-B2 | 54,9 | +5,5 |
| **Nhóm** (3 seed) | SegFormer-B3 | 63,4 | **+0,8** |

- Điểm tuyệt đối khác nhau do mỗi bài một cấu hình → **chỉ so mức tăng**.
- **Mọi bài chấm lại đều chỉ được 0 đến +5,5**, không bài nào gần +12,1. Bài cùng dùng B3 (SSVIF) ra +0,03, rất gần nhóm.
- Mạng chấm càng mạnh, SAGE càng ít lợi (B2: +3,7…+5,5; B3/B5: 0…+1). Nhóm cũng thấy điều này: SegFormer-B0 +1,7 (chạy thử riêng của Mân, 2 seed, chưa đưa lên repo) → B3 +0,8.
- **Điều nhóm làm thêm mà các bài trên không có:** ghi đủ cấu hình, nhiều seed, khoảng tin cậy, và đối chứng ghép trung bình.

---

## 7. SAGE thắng / thua ở đâu?

Theo **từng ảnh** (273 ảnh, cộng 3 seed; thắng/thua = chênh mIoU ảnh > 1 điểm): SAGE thắng VIS **159 ảnh**, thua **51**, hòa 63. Ảnh tối: thắng 72, chỉ thua 7. Ảnh sáng: thắng 35, thua 22, hòa 34.
*(Tính theo từng ảnh thì vật nhỏ như người được coi trọng hơn, nên SAGE trông tốt hơn (TB +2,1 mỗi ảnh); tính gộp cả tập thì các vùng lớn chi phối nên chỉ +0,8.)*
Trong các ảnh chênh nhiều (> 3 điểm), lớp quyết định là **người** ở 68 ảnh SAGE thắng, chỉ 6 ảnh SAGE thua.

| | Khi nào | Ví dụ |
|---|---|---|
| **Thắng** | Ảnh thường mất thông tin: **đêm, khói, đèn chói** – gần như toàn bộ nhờ **lớp người** | 00089 (khói che người: ảnh thường đoán người thành cây, SAGE đoán đúng, IoU người 0 → 85) |
| **Thua** | **Ban ngày**, **xe lớn** (xe buýt, xe tải bị đoán thành xe con – ảnh nhiệt làm mất chữ, màu), mất đèn giao thông | 00388 (xe buýt → xe con), 00167 (ban ngày, mất đèn giao thông) |

Chi tiết: [`ca_thang_thua.txt`](ca_thang_thua.txt). Hình (dự đoán seed 0): [`sage_thang.png`](sage_thang.png), [`sage_thua.png`](sage_thua.png).

---

## 8. Đã kiểm lỗi những gì

Vì kết quả lệch bài, nhóm đã kiểm kỹ trước khi kết luận – **không thấy lỗi**:
- Danh sách ảnh train / test không giao nhau; md5 ảnh train và test không trùng.
- Chấm lại độc lập từ ảnh dự đoán + nhãn gốc → khớp 100% số cũ.
- Log Kaggle: đủ 1.500 ảnh mỗi loại, loss hội tụ, đủ 15.200 vòng lặp.
- Ba seed cho kết quả cùng chiều (SAGE − VIS: +0,55 / +0,51 / +1,40).

---

## 9. Kết luận, nhận định và ý nghĩa cho đồ án

### 9.1. Kết luận (có số liệu, 3 seed, khoảng tin cậy)

1. **Với checkpoint SAGE công bố, nhóm không tái lập được mức tăng +12,1 của bài.** SAGE − VIS chỉ **+0,8** (gộp 3 seed, CI [−0,86; +3,04]), không seed nào có ý nghĩa thống kê. Theo tiêu chí đặt trước (≤ +4 là không tái lập được) → **không tái lập được**.
2. **Ghép trung bình (TB) tốt hơn SAGE, nhưng sát ngưỡng.** Quy tắc đặt trước (ghi 13/11, trước khi có seed 2): TB > SAGE ở cả 3 seed **và** gộp 3 seed có ý nghĩa → viết "tốt hơn". Kết quả: TB > SAGE cả 3 seed (+1,65 / +0,84 / +0,67), gộp **+1,05** [+0,08; +2,16] → **đạt**. Khi trình bày nói rõ mức chênh nhỏ (~1 mIoU) và cận dưới khoảng tin cậy sát 0.
3. **Ghép tốt hơn hẳn chỉ dùng ảnh nhiệt:** SAGE − IR **+6,96** [+3,58; +10,46], có ý nghĩa ở cả 3 seed.
4. **Bước tạo ảnh SAGE của nhóm đúng:** checkpoint công bố chạy ra gần trùng ảnh mẫu của tác giả (lệch trung bình 0,2 mức xám, tương quan 0,9999).

### 9.2. Nhận định (suy ra từ số liệu, chưa kiểm chứng trực tiếp)

1. **Chênh lệch với bài có thể nằm ở ảnh SAGE bài đã dùng** (ví dụ ảnh từ Teacher, hoặc từ checkpoint khác). Căn cứ: VIS khớp bài, chỉ SAGE lệch. Chưa chắc vì bài không ghi Bảng 1 dùng Teacher hay Student, checkpoint nào; IR cũng không khớp hẳn nên chưa loại trừ khác biệt cách chấm.
2. **Checkpoint công bố nhiều khả năng được train trên FMB.** Căn cứ gián tiếp: thống kê BatchNorm khớp FMB train nhất; chỉ phản ánh giai đoạn cuối quá trình train.
3. **Cái lợi ở lớp người đến từ ảnh nhiệt, chưa thấy là nhờ SAM.** Căn cứ: IR và TB nhận người tốt ngang SAGE (~70 so với VIS ~59). Chưa có thí nghiệm bỏ SAM để kiểm.
4. **SAGE mạnh ở ảnh tối và lớp người, yếu ở xe lớn và ban ngày** (mục 7). Đây là mô tả theo từng ảnh, không có tiêu chí đặt trước; ở ảnh tối TB − SAGE +3,4 nhưng chưa có ý nghĩa (91 ảnh); lớp xe buýt dao động mạnh giữa seed.

### 9.3. Ý nghĩa cho đồ án

- **Gap đề xuất (chưa chốt):** *lợi ích phân đoạn của SAGE và phần SAM tốn kém của nó chưa được kiểm chứng chắc.* Bước tiếp theo kiểm trực tiếp: train lại Student **bỏ SAM** rồi chấm bằng đúng thiết lập này (đã chuẩn bị code, chưa chạy, chưa đưa lên repo).
- **Đề xuất:** dùng thiết lập B3 này làm **baseline chung của nhóm** – mọi cải tiến sau này phải hơn **cả SAGE lẫn TB** trên đúng thiết lập này (cùng split, cùng cấu hình, cùng code chấm, 280 + 273 ảnh, nhiều seed).

---

## 10. Giới hạn

- **3 seed**; khoảng tin cậy gộp chưa tính dao động giữa seed; chỉ **một bộ dữ liệu (FMB)** – chưa biết kết luận có giữ trên bộ khác (ứng viên: MFNet, có nhãn phân đoạn).
- Một số chi tiết của bài phải đoán (khởi tạo, cách tính mIoU…). Đã kiểm 7 lớp in ra để giảm ảnh hưởng, nhưng không loại trừ hết.
- Lớp hiếm (xe buýt, xe tải, xe máy) dao động mạnh giữa seed.
- Chưa chấm bằng X-Decoder không train (cách chấm thứ hai trong Bảng 1). Theo bài, cách chấm này cho SAGE chỉ hơn VIS +1,0 (50,1 → 51,1), cùng chiều với kết quả B3 của nhóm.

---

## 11. File và cách chạy lại

### 11.1. Trong repo

| Cái gì | Ở đâu |
|---|---|
| Code (xuất ảnh, train, chấm, tổng hợp, khoảng tin cậy, ca thắng/thua) | [`src/segmentation/`](../../../src/segmentation/README.md) |
| Danh sách ảnh (split) | [`data/manifests/`](../../../data/manifests/README.md) – đúng 3 file nhóm đang dùng (train 1.045 + dev 175 gộp lại thành 1.220 ảnh train; test 280) |
| Notebook Kaggle | [`notebooks/man/sage_b3_kaggle.ipynb`](../../../notebooks/man/sage_b3_kaggle.ipynb) – một notebook, chọn `BIEN_THE` và `SEED` ở ô đầu |
| Kết quả nhỏ của từng run | `runs/<run>/`: `config.json` (cấu hình), `log.csv` (loss theo vòng lặp), `test_ket_qua_last.json` (mIoU 280/273 ảnh), `test_iou_lop_last.csv` (IoU từng lớp) |
| Bảng tổng hợp | [`tong_hop.csv`](tong_hop.csv), [`tong_hop.txt`](tong_hop.txt) – mọi quy ước mIoU, IoU từng lớp, so với bài |
| Chênh lệch + khoảng tin cậy | [`so_sanh_b3.txt`](so_sanh_b3.txt) – từng seed, gộp 3 seed, chia tối/vừa/sáng |
| Danh mục artifact ngoài Git | [`artifact_manifest.json`](artifact_manifest.json) |

### 11.2. Ngoài Git (Kaggle, riêng tư – xin quyền Mân)

| Dataset | Nội dung |
|---|---|
| `thanhmn123/sage-b3-fmb-checkpoints` | 12 `last.pt` (189 MB mỗi file) + `config.json` |
| `thanhmn123/sage-b3-fmb-ketqua` | 12 run: ma trận nhầm lẫn từng ảnh `test_nham_last.npz`, nhãn dự đoán `pred_last/`, log, json |
| `thanhmn123/fmb-ivif-sage` | Dữ liệu FMB (train + test, không có thư mục `color`) |

Notebook đã chạy: `thanhmn123/sage-b3-fmb` (VIS + SAGE; v2 seed 0, v3 seed 1, v4 seed 2) và `thanhmn123/sage-b3-fmb-tb-ir` (TB + IR; v1 seed 0, v2 seed 1, v3 seed 2). Mỗi run train ~2,6 giờ trên T4.

**Checkpoint (`last.pt`; md5 12 ký tự đầu, md5 đầy đủ trong `artifact_manifest.json`):**

| Biến thể | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| VIS | `1785b91752cd` | `4a46786b2234` | `c482d5e6c0e6` |
| IR | `68512c087fd2` | `f1fd66f264b8` | `a8f19c4616fd` |
| SAGE | `c736af3a0f55` | `e62e43a87c36` | `63b1b453ddf8` |
| TB | `0e8578e93b3b` | `c58ed79cc077` | `cec18cf7270b` |

### 11.3. Chạy lại

Tính lại bảng và khoảng tin cậy (CPU, vài phút) – tải dataset `sage-b3-fmb-ketqua` về, ví dụ vào `runs/b3_fmb_seg/`:

```powershell
$env:SAGE_RUNS = "runs/b3_fmb_seg/runs"
python src/segmentation/tong_hop.py
python src/segmentation/so_sanh_b3.py > so_sanh_b3.txt
```

Đã kiểm 14/11: chạy từ repo ra `tong_hop.txt`, `so_sanh_b3.txt`, `ca_thang_thua.txt` khớp từng dòng với bản gốc.

Train lại một run: chạy notebook Kaggle (GPU T4) hoặc trên máy có GPU – xem [`src/segmentation/README.md`](../../../src/segmentation/README.md).
