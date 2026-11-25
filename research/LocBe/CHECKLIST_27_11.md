# Checklist theo yêu cầu nhóm — Lộc Bé — cập nhật 26/11/2026

| Mục | Trạng thái | Bằng chứng / việc còn thiếu |
|---|---|---|
| Method card & 3 references | DONE | `METHOD_SUMMARY.md` |
| Frozen code/config/SHA | DONE (không thay đổi công thức) | `source/`, `configs/`, `FROZEN_MANIFEST.json` |
| Train/Dev/Test manifests & provenance | PARTIAL | ID/hash lưu trước; FMB gốc và SAGE pretrained còn ở Kaggle Dataset, chưa đóng gói Drive |
| Self-test & confusion parity | DONE | `results/evaluation.json`, `evidence/inference_only/report/confusion_parity.json` PASS; seed 0 replay 280/280 images, 0 mismatch |
| Per-seed mIoU mean±SD, baseline SAGE/TB | DONE | `results/results_overall.csv` |
| Per-class IoU và delta | DONE | `results/results_per_class.csv`, `RESULTS_DISCUSSION.md` |
| Raw per-image confusion all 3 seeds | AVAILABLE LOCAL / GITIGNORE | `results/confusion_per_image.npz`, `evidence/seed*_test_nham_last.npz` có trong gói cập nhật; muốn commit phải Git `add -f` sau khi được nhóm chấp nhận |
| 280 C3-T prediction PNG seed 0 | DONE LOCAL, AWAIT PUSH | `evidence/inference_only/predictions/c3t_seed0/`; 280/280 SHA256 khớp manifest, 600×800 class IDs |
| 4–6 qualitative panels VIS/IR/GT/fusion/pred | PARTIAL | Đã có 6 panel thật `figures/qualitative/`; còn thiếu **SAGE B3 prediction** nên chưa minh hoạ được C3-T thắng/thua trực tiếp so SAGE |
| C3-T pure CPU latency / B3 GPU latency & VRAM | PARTIAL | 93.67 ms C3-T CPU (không I/O/SAGE); B3 seed0 72.96 ms T4 (forward+transfer), peak 1139.6 MiB allocated; **end-to-end SAGE→C3T→B3 chưa đo** |
| Fusion EN/SD/SCD | DONE (theo định nghĩa đã công bố) | `results/fusion_metrics_summary.csv`, 280/273 ảnh, cả SAGE/TB/C3-T |
| Fusion MS-SSIM | NOT DONE | Chưa khóa quy ước tham chiếu hai nguồn VIS/IR; không tự suy diễn số liệu |
| Baseline+TTA | NOT DONE / IF APPLICABLE | Cần cùng seed, weights và phép biến đổi được khóa |
| C2→C3-T matched ablation | PARTIAL | Exploratory Dev trước đây; chưa có paired official protocol |
| FMB Test protocol | DOCUMENTED | **POST-EXPOSURE / EXPLORATORY**, không gọi kiểm định độc lập |
| MFNet independent dataset | OPTIONAL / NOT DONE | Cần B3 riêng cho 9 lớp MFNet, không dùng trực tiếp FMB 15 lớp |
| B3 checkpoints 3×last.pt | VERIFIED EXTERNAL | SHA256 3/3 khớp `FROZEN_MANIFEST.json`; file nhị phân 189MB/seed ở Kaggle Output/local, cần lưu Drive nhóm và cấp link |
| Git branch LocBe | DONE INITIAL PUSH | `5a609d275fffa4234a6d061c0eab6b796424ca6b`; **bằng chứng ngày 26/11 chưa push** |
| Final ZIP/Drive + checksum | NEED REPACKAGE | Sau khi upload Drive/push bổ sung, tạo ZIP cuối và SHA256 mới, không dùng ZIP trước khi bổ sung làm bản cuối |

**Không tối ưu theo Test đã xem; không trình bày missing/partial như done.**
