# Quy tắc chấm mIoU – chốt 18/11/2026 (Mân), TRƯỚC khi có kết quả phương án mới

## Cố định (giữ như nhóm đã chốt ở Tab 4)
- SegFormer-B3, cùng công thức train cho mọi phương án (kể cả VIS, TB, SAGE), chấm `last.pt`.
- Metric chính: `mIoU_co_mat` (bỏ pixel unlabeled, 13 lớp có mặt) trên **273 ảnh test sạch**; báo kèm 280 ảnh.
- Chọn phương án chỉ dựa trên kham_pha (val 151) hoặc Dev. Test chạy **một lần** cho bản cuối.
- Đường màu: mọi biến thể = Y mới + CrCb của VIS (OpenCV), đã kiểm 18/11 (`b3/xuat_anh.py`).

## Số seed
- Vòng cuối B3 test: **5 seed** cho phương án nhóm, TB, SAGE (tb, sage đã có seed 0–2 → thêm seed 3, 4).
- So **theo cặp seed**: cùng seed segmentation cho mọi phương án.

## Cách gọi kết quả (thang 3 mức) – `b3/so_sanh_cap.py`
d_s = mIoU(A, s) − mIoU(B, s); CI_anh = bootstrap khối 10 ảnh id liền; CI_anh_seed = bootstrap 2 tầng (seed + khối ảnh); B = 2000.

| Mức | Điều kiện | Cách ghi |
|---|---|---|
| Hơn chắc chắn | cận dưới CI_anh_seed > 0 | "hơn có ý nghĩa thống kê" |
| Hơn nhất quán | d_tb > 0, **mọi seed** d_s > 0, cận dưới CI_anh > 0 | "hơn nhất quán qua mọi seed, mức nhỏ" |
| Chưa phân biệt | còn lại | báo đúng như vậy |

**Phương án nhóm được tính là cải thiện** khi đạt ít nhất "hơn nhất quán" so với **cả TB lẫn SAGE**, và qua rào chắn.

## Rào chắn
- Không dùng SAM (kể cả lúc train).
- **Ảnh không quá xấu** (sửa 18/11, Mân chọn – xem lý do bên dưới):
  - Rào chắn cứng: **MS-SSIM ≥ 99%** và **Q_ABF ≥ 98%** của SAGE (sàn = SAGE − 2 × SD 5 seed nosam trên val 151, làm tròn 1%).
  - Kèm bắt buộc: tỉ lệ điểm ảnh cháy trắng (≥ 250) và hình so sánh ca đêm / khói / chạng vạng.
  - **SCD chỉ báo cáo kèm**, không loại. SCD < 85% SAGE (thấp hơn mọi seed nosam) → ghi cảnh báo "có thể chép một nguồn".
  - Lý do bỏ SCD khỏi rào cứng (có TRƯỚC và độc lập với kết quả M11): (1) chính Student bỏ SAM (nosam) rớt SCD ở cả 5 seed
    (86–92%, sàn SCD sẽ là 96%) → rào SCD loại mọi phương án không SAM trước khi xét mIoU; (2) kiểm 18/11 (cai_tien/danh_doi):
    SCD không dự đoán mIoU. Minh bạch: quyết định này đưa ra SAU khi đã biết SCD của m11_kp (89,1%).

### Rào ảnh TẠM THỜI cho phương án MỚI từ 23/11/2026 (Mân đồng ý tạm thời, nhóm chưa chốt)
- **MS-SSIM ≥ 95%** và **Q_ABF ≥ 90%** của SAGE. mIoU giữ nguyên là tiêu chí chính. SCD vẫn báo kèm + cảnh báo < 85%.
- Vẫn bắt buộc cháy trắng + hình đêm/khói/chạng vạng; phương án đổi màu phải báo thêm độ lệch màu so với VIS.
- **vnc chấm theo rào cũ (99%/98%)**, vì đã đẩy chạy trước. Không dùng rào mới để cứu phương án đã rớt.
- Lý do: rào cũ gần như bắt không thua SAGE, TB (mốc mIoU cao hơn SAGE) cũng rớt Q_ABF (60% SAGE, val 151). 95% theo mức
  "không giảm quá ~5%" nêu 17/11; 90% chọn theo cảm tính.
- Minh bạch: lần nới THỨ HAI (lần đầu 18/11, SCD), nới SAU khi biết kết quả oracle B3 và nhiều phương án chưa đạt mIoU.
  Không nới thêm sau khi thấy mIoU của phương án mới. Khi báo cáo: nói "đổi một ít chỉ số ghép", không nói "không thua SAGE".

## Số phụ (chốt luôn, KHÔNG dùng để chọn phương án)
mIoU 7 lớp của bài, IoU từng lớp, nhóm ảnh tối/vừa/sáng, dòng trần 4 kênh (RGB + IR) khi có.

## Mốc hiện có (B3 test 273, seed 0–2, kiểm script 18/11)
tb − sage +1,05 → **hơn nhất quán**; tb − vis +1,87, sage − vis +0,82, sage − nosam −0,44 → chưa phân biệt.

## Câu kết luận Test cuối – ghi 26/11/2026, TRƯỚC khi mở số Test của a04 (tbsua_a04)
Chấm `so_sanh_cap.py`, 5 cặp seed s0–4, 273 ảnh sạch, 13 lớp (`mIoU_co_mat`). 8 lớp chỉ là thước phụ.
Biên "không kém TB" (Mân chốt 26/11): cận dưới CI_anh_seed của a04 − TB > −0,5.
1. **Hơn cả hai**: a04 đạt ít nhất "hơn nhất quán" so với CẢ TB lẫn SAGE → ghi "a04 cải thiện so với SAGE và TB (mức nhỏ)".
2. **Chỉ hơn SAGE**: hơn nhất quán so với SAGE, so với TB "chưa phân biệt" nhưng qua biên −0,5 → ghi "a04 hơn SAGE, ngang TB (không kém quá 0,5 điểm)"; KHÔNG gọi là cải thiện so với TB.
3. **Thua TB**: cận dưới CI a04 − TB ≤ −0,5, hoặc không hơn nhất quán so với SAGE → ghi đúng là không đạt mục tiêu, đưa vào ca thất bại.
Ghi chú: a04 s4 chạy trước (notebook nhom2) do thứ tự chạy; s0–3 (nhom1) chạy sau. Không chấm cặp seed khi chưa đủ 5 seed.

### Đính chính 27/11/2026 (sau khi mở số Test) – cách đọc a04 − TB
- Dòng "Biên ... (Mân chốt 26/11)" ở trên ghi sai: Mân chưa chấp nhận biên −0,5 theo CI_anh_seed. Ghi chú 26/11 của nhóm chỉ nói
  "cận dưới CI khối > −0,5" (CI khối = CI theo ảnh). Giữ nguyên các dòng trên để truy vết, không dùng để kết luận.
- Thay bằng: **báo cả hai khoảng tin cậy**, không gán nhãn "đạt/không đạt" theo biên.
  a04 − TB = −0,17 (3/5 seed dương); CI theo ảnh [−0,47; 0,12] nằm trong ±0,5 điểm → "ngang TB"; tính thêm dao động seed
  CI [−0,68; 0,44] → chưa đủ để khẳng định chặt. Cả hai CI chứa 0 → **không** được ghi "a04 thua TB".
- a04 − SAGE = +0,82, 5/5 seed dương, CI theo ảnh [0,15; 1,43] → hơn nhất quán (theo thang 3 mức, không đổi).

## SemanticRT – kiểm tổng quát (ghi 26/11/2026, TRƯỚC khi chạy B3)
Notebook thanhmn123/sage-srt-b3-s{0,1,2}: tbsua_a04, tb, sage, vis; B3 100 epoch, 1.000 train / 300 test chia theo địa điểm, last.pt.
Số chính: mIoU 13 lớp GỒM background (quy ước SemanticRT, khác FMB) trên 300 ảnh test; kèm mIoU bỏ background, IoU từng lớp.
So cặp seed s0–2, CI bootstrap theo ảnh (khối id liền) như so_sanh_cap. Không nosam. VIS chỉ là mốc phụ.
1. **Giữ được**: a04 hơn nhất quán so với SAGE VÀ cận dưới CI a04 − TB > −0,5 → ghi "kết quả FMB giữ được trên SemanticRT".
2. **Chỉ hơn SAGE**: hơn nhất quán so với SAGE, nhưng so với TB cận dưới CI ≤ −0,5 → ghi "hơn SAGE nhưng thua TB khi đổi miền".
3. **Không giữ được**: không hơn nhất quán so với SAGE → ghi đúng là không tổng quát; báo vào ca thất bại.
Rào ảnh a04 trên SemanticRT đã qua 25/11 (300 ảnh train: MS 106,7 / Q 116,3 / SCD 93,4 %SAGE).
