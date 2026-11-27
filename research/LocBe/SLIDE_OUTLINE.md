# Gợi ý 6 slide cho phần trình bày của Lộc Bé (dựa vào tài liệu nhóm 25/11)

> Đây là **dàn ý**, chưa phải tệp PowerPoint. Chỉ đưa hình và kết quả đã có bằng chứng.

1. **Phương pháp và bài toán** — ảnh VIS+IR → Fusion RGB → SegFormer-B3; lý do chọn C3-T, giả thuyết giữ biên IR.
2. **Pipeline và công thức** — SAGE → C2 edge residual (alpha=0.20) → Thermal-only Guard (threshold=0.30). Sơ đồ bằng các box/text, không đổi công thức.
3. **Thiết lập và freeze** — FMB 1220 train/280 Test/273 clean; 3 seed; 100 epochs; last.pt; checksum group, SAGE, model; note test `post-exposure / exploratory`.
4. **Kết quả định lượng** — biểu đồ `figures/quant_miou_by_seed.png`; mIoU13 C3T=63.1994±0.4631, SAGE=63.37, TB=64.4233; kết luận không vượt TB.
5. **Phân tích lớp** — `figures/quant_class_delta_vs_sage.png` + `RESULTS_DISCUSSION.md`; ba lớp tăng tốt nhất, ba lớp giảm nhiều nhất; mọi lý giải chỉ là giả thuyết.
6. **Minh họa lỗi, trade-offs, giới hạn** — các panel VIS/IR/GT/pred bắt buộc đang thiếu, cần khôi phục checkpoint và pred PNG; chỉ ghi nhận chi phí export đo được; giới hạn TTA, fusion metrics, MFNet chưa đánh giá.

**Lời kết ngắn:** "Em đã triển khai và khóa phương pháp DCTR-C3-T, hoàn tất ba seed B3 và kiểm chứng lại confusion matrix. Phương pháp không vượt được TB trên Test sạch. Em bàn giao kết quả âm, mã nguồn, checksum và các giới hạn; không tối ưu tiếp theo Test."
