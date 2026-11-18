# Bảng kiểm chứng Metric — Lộc lớn (Evaluation)

Thư mục này lưu trữ các bảng tra cứu, kiểm chứng toán học và đối chuẩn cho **8 chỉ số cốt lõi bắt buộc** theo đúng đồ thị violin trong bài báo **SAGE (CVPR 2025, Figure 5)**, kèm các metric bổ trợ chẩn đoán:
- **Nhóm 1 — No-Reference Image Quality Assessment (NR-IQA)** (đánh giá trực tiếp ảnh fused $F$ không cần ảnh chuẩn):
  1. `PaQ-2-PiQ↑` (tỷ lệ x1.0)
  2. `MUSIQ↑` (tỷ lệ x1.0)
  3. `BRISQUE↓` (tỷ lệ x1.0, càng thấp càng tốt)
  4. `NI-QE*10↓` (tỷ lệ x10.0, càng thấp càng tốt)
- **Nhóm 2 — Infrared-Visible Fusion Fidelity** (đo độ trung thực, lượng thông tin và bảo toàn cấu trúc so với ảnh hồng ngoại $IR$ và khả kiến $VIS$):
  5. `EN*5↑` (tỷ lệ x5.0)
  6. `SD*0.5↑` (tỷ lệ x0.5)
  7. `SCD*20↑` (tỷ lệ x20.0)
  8. `MS-SSIM*20↑` (tỷ lệ x20.0)

---

## 1. Danh sách bảng

| File | Nội dung chính |
|---|---|
| [`metric_definitions_reference.csv`](metric_definitions_reference.csv) | Định nghĩa toán học, citation gốc, miền giá trị lý thuyết, hệ số nhân đồ thị và trạng thái kiểm chứng cho toàn bộ metric. |
| [`sage_paper_8metrics_benchmark.csv`](sage_paper_8metrics_benchmark.csv) | Bảng quy cách 8 metric chuẩn mực đối chiếu theo đồ thị violin của bài báo SAGE (CVPR 2025 Figure 5) so sánh cùng U2Fusion, TarDAL, SegMif, FILM, SHIP, MRFS, EMMA, TIMFusion. |
| [`reference_cases_verification.csv`](reference_cases_verification.csv) | Kết quả kiểm chứng trên 8 ca đối chứng nhân tạo (nhiễu đều, ảnh hằng, F=IR, F=255-IR, dịch độ sáng, v.v.). |
| [`aggregation_comparison.csv`](aggregation_comparison.csv) | So sánh 3 phương thức tổng hợp MS-SSIM đa nguồn: Equal Mean (0.5/0.5), Direct Sum (1.0/1.0), và Variance-Weighted. |
| [`loc_be_parity_verification.csv`](loc_be_parity_verification.csv) | Bảng đối chuẩn xác nhận tính tương đồng (parity) với các số liệu 10 ảnh của Lộc bé (SAGE, PCDFusion, Y_AVG, Y_MAX). |

---

## 2. Ý nghĩa & Quy cách 8 Metric trong bài báo SAGE (CVPR 2025)

Tại sao bài báo SAGE lại chia thành 2 bảng violin gồm đúng 8 metric này?
1. **Các độ đo kết hợp truyền thống (EN, SD, SCD, MS-SSIM):**
   - Đánh giá khả năng chuyển giao thông tin từ 2 nguồn ảnh ($IR, VIS$) vào $F$.
   - Tuy nhiên, các độ đo này **không phát hiện được hiện tượng ảo giác (hallucination artifacts)** hoặc mất tự nhiên về mặt thị giác con người khi mô hình sinh ra pixel không hợp lý.
2. **Các độ đo chất lượng ảnh mù / không cần tham chiếu (PaQ-2-PiQ, MUSIQ, BRISQUE, NIQE):**
   - **`BRISQUE`** và **`NIQE`**: Dựa trên thống kê cảnh tự nhiên (Natural Scene Statistics — NSS) qua hệ số MSCN và phân phối AGGD. Nếu ảnh hợp nhất bị giả tạo, biến dạng biên hoặc mất độ mượt tự nhiên, điểm BRISQUE và NIQE sẽ tăng vọt (xấu đi).
   - **`PaQ-2-PiQ`** và **`MUSIQ`**: Dùng mạng nơ-ron sâu và Transformer đa tỷ lệ được huấn luyện trên hàng triệu đánh giá chủ quan của con người (LIVE-FB, KonIQ-10k). Các mô hình này phản ánh chính xác cảm nhận thị giác thực tế của con người về độ sắc nét, độ tương phản và tính chân thực.
3. **Ý nghĩa các hệ số co giãn trên đồ thị violin:**
   - Để vẽ chung trên cùng trục tung $[10, 90]$ một cách trực quan và cân đối, tác giả SAGE đã chuẩn hóa tỷ lệ hiển thị:
     - `NI-QE * 10`: điểm gốc từ ~3 đến ~8 đưa về ~30 đến ~80.
     - `EN * 5`: entropy gốc từ ~6.5 đến ~7.5 bits đưa về ~32.5 đến ~37.5.
     - `SD * 0.5`: độ lệch chuẩn gốc từ ~40 đến ~80 đưa về ~20 đến ~40.
     - `SCD * 20`: hiệu tương quan gốc từ ~1.2 đến ~1.8 đưa về ~24 đến ~36.
     - `MS-SSIM * 20`: cấu trúc đa tỷ lệ gốc từ ~0.80 đến ~0.95 đưa về ~16 đến ~19.
