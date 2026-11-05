# Quy tắc làm việc nhóm

## 1. Owner theo module

| Thành viên | Trách nhiệm chính | Đường dẫn chính |
|---|---|---|
| Quân | Baseline SAGE Student, inference FMB, checkpoint config, integration | `notebooks/quan/`, `src/baseline/`, `configs/baseline/`, `scripts/` |
| Mân | Audit FMB, manifest/split, chống leakage, data/robustness problem | `notebooks/man/`, `src/data/`, `data/manifests/`, `data/audit/` |
| Lộc bé | Limitation/ablation/related work, research proposal, prototype | `notebooks/loc_be/`, `src/analysis/`, `src/methods/` |
| Lộc lớn | EN/SD/SCD/MS-SSIM, formula verification, comparison/ablation | `notebooks/loc_lon/`, `src/evaluation/`, `tests/`, `reports/tables/` |

Owner là người review bắt buộc khi code thuộc module của họ thay đổi, không phải người duy nhất
được đóng góp. Không push trực tiếp vào branch của người khác.

## 2. Mười quy tắc bắt buộc

1. `main` là branch tích hợp ổn định; sau commit scaffold không commit trực tiếp vào `main`.
2. Bắt đầu bằng branch riêng: `feature/quan-baseline`, `feature/man-dataset`,
   `feature/locbe-research`, `feature/loclon-evaluation`. Tác vụ sau dùng branch ngắn hạn rõ nghĩa.
3. Không push vào branch của người khác.
4. Mọi thay đổi vào `main` qua Pull Request và ít nhất một người khác review.
5. PR phải ghi thay đổi, file, cách chạy, output thật, validation, test và ảnh hưởng baseline.
6. Không commit dataset lớn, `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, archive, hàng trăm ảnh
   output hay notebook có output quá lớn.
7. Không lưu password, API token hoặc Kaggle credential. Nếu lỡ commit, báo ngay để revoke;
   xóa file khỏi commit không làm token cũ an toàn trở lại.
8. Experiment phải ghi seed khi phù hợp, checkpoint, manifest, preprocessing, environment và
   commit hash.
9. Không ghi đè/xóa kết quả cũ; mỗi experiment có `run_id` duy nhất.
10. Không đổi test set, evaluation protocol hoặc primary metric sau khi xem kết quả nếu chưa có
    quyết định nhóm được ghi lại và báo cáo minh bạch.

## 3. Definition of Done

- Code/config/document ở đúng owner path và có liên kết issue/card khi phù hợp.
- Có lệnh chạy, input/output contract và kết quả kiểm tra thật.
- CPU unit test cho logic có thể test; GPU test ghi log và môi trường ngoài CI.
- Không có dữ liệu/checkpoint/secret/output lớn trong Git.
- Reviewer kiểm tra claim: “chạy được” khác với “đúng paper”, “pretrained inference” khác với
  “reproduction full training”.

## 4. Quyền repository

Không giả định mọi người đã có write access. Chủ repository phải mời đúng GitHub username tại
**Settings → Collaborators and teams → Add people**. Thành viên xác nhận lời mời trước khi push.
Nếu chưa có quyền, dùng fork/patch theo quyết định của chủ repo; không chia sẻ token cá nhân.
