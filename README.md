# SAGE Research – Infrared-Visible Image Fusion

Repository đồ án nghiên cứu của nhóm 4 người về hợp nhất ảnh hồng ngoại–khả kiến, dựa trên
paper CVPR 2025 **Every SAM Drop Counts: Embracing Semantic Priors for Multi-Modality Image
Fusion and Beyond** và [source SAGE_IVIF chính thức](https://github.com/RollingPlain/SAGE_IVIF).

Mục tiêu là tái lập **pretrained Student inference** trên FMB, xây evaluation pipeline được kiểm
chứng và thử tối đa 1–2 cải tiến/robustness direction phù hợp Kaggle GPU. Chạy pretrained Student
không đồng nghĩa tái lập full training hoặc downstream mIoU của paper.

> **Trạng thái 05/11/2026:** scaffold quản lý nghiên cứu. Repo chưa chứa model, dataset,
> checkpoint, notebook bàn giao hay kết quả thí nghiệm thật. Deadline: **17/12/2026**.

## Nhóm và ownership

| Thành viên | Phần chính | Vị trí |
|---|---|---|
| Quân | Baseline & Integration | `notebooks/quan/`, `src/baseline/`, `configs/baseline/`, `scripts/` |
| Mân | Dataset & Protocol | `notebooks/man/`, `src/data/`, `data/manifests/`, `data/audit/` |
| Lộc bé | Research Method | `notebooks/loc_be/`, `src/analysis/`, `src/methods/` |
| Lộc lớn | Evaluation | `notebooks/loc_lon/`, `src/evaluation/`, `tests/`, `reports/tables/` |

Ownership yêu cầu owner review, không cấm thành viên khác đóng góp. Xem trạng thái bàn giao thật
tại [`docs/TEAM_TASKS.md`](docs/TEAM_TASKS.md).

## Cấu trúc

```text
.
├── .github/          # CI, issue và PR templates
├── configs/          # baseline/evaluation templates và config run
├── data/             # README, manifest/audit nhỏ; raw data bị ignore
├── external/         # upstream pin theo URL/commit/license
├── checkpoints/      # vị trí local; artifact bị ignore
├── notebooks/        # workspace theo từng thành viên
├── src/              # code do nhóm tích hợp/duy trì
├── scripts/          # validation và entry point đã kiểm chứng
├── tests/            # CPU-only tests
├── runs/             # run theo output contract; artifact bị ignore
├── outputs/          # output khám phá cục bộ, không phải record chính thức
├── docs/             # protocol, workflow, cards và meeting notes
└── reports/          # bảng/hình/báo cáo nhỏ được review
```

`external/` chứa upstream bên ngoài được pin và ghi license; `src/` chứa integration, evaluator
và phương pháp do nhóm bảo trì. Không copy mã upstream âm thầm vào `src/`.

## 1. Clone và cài môi trường

Chủ repo cần vào **GitHub → Settings → Collaborators and teams** và mời **đúng GitHub username**
của từng thành viên. Không mặc định display name là username hoặc mọi người đã có quyền ghi.

```powershell
git clone https://github.com/quan14333/xulianh.git
Set-Location xulianh
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

## 2. Tạo branch cá nhân

Chỉ chạy lệnh tương ứng với mình:

```powershell
git switch -c feature/quan-baseline
git switch -c feature/man-dataset
git switch -c feature/locbe-research
git switch -c feature/loclon-evaluation
```

Sau đó push bằng `git push -u origin <ten-branch>`. Không push vào branch người khác và không
commit trực tiếp `main` sau scaffold. Quy trình đầy đủ và xử lý conflict:
[`docs/GIT_WORKFLOW.md`](docs/GIT_WORKFLOW.md).

## 3. Đặt FMB và checkpoint

```text
data/raw/FMB/                 # IR/VIS/labels local, không commit
data/manifests/               # CSV nhỏ đã xác minh, được commit
data/audit/                   # audit script/report nhỏ
checkpoints/pretrained/       # checkpoint local, không commit
```

Không commit `.pt/.pth/.ckpt`, archive, Kaggle credential hay dataset. Cần xác minh khác biệt
800×600 và 640×512 cùng RGB/YCbCr/IR preprocessing trước baseline. Xem
[`docs/DATA_POLICY.md`](docs/DATA_POLICY.md).

## 4. Kaggle và experiment output

Kaggle dùng `/kaggle/input` cho dataset/checkpoint read-only và `/kaggle/working` cho output.
Mỗi run có ID duy nhất, config, environment, manifest, log và fused/metrics tùy loại. Artifact lớn
lưu bằng Kaggle Output/Storage của nhóm; Git chỉ lưu config, manifest, tài liệu, bảng nhỏ và hướng
dẫn lấy artifact. Không ghi đè run cũ. Contract chi tiết:
[`docs/OUTPUT_CONTRACT.md`](docs/OUTPUT_CONTRACT.md).

## 5. Validation local

```powershell
python -m compileall -q src scripts tests
python scripts/validate_repo.py
python -m pytest
```

CI chạy ba kiểm tra này trên Pull Request và push vào `main`, không tải FMB, SAM checkpoint hay
GPU dependency.

## 6. Nộp code qua Pull Request

```powershell
git status
git add <duong-dan-cu-the>
git diff --cached
git commit -m "feat: mo ta thay doi"
git push -u origin <ten-branch>
```

Mở PR vào `main`, điền template, đính run/output thật, chờ CI và ít nhất 1 người khác review.
Sửa module của owner khác thì yêu cầu owner đó review. Không ghi kết quả dự kiến thành kết quả thật.

## Tài liệu bắt buộc đọc

- [Quy tắc nhóm](docs/TEAM_RULES.md)
- [Git workflow](docs/GIT_WORKFLOW.md)
- [Data policy](docs/DATA_POLICY.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Output contract](docs/OUTPUT_CONTRACT.md)
- [Lịch họp 3 ngày/lần](docs/MEETING_SCHEDULE.md)
- [Problem–Solution Card](docs/problem_cards/TEMPLATE.md)

Repository chưa có LICENSE vì chủ sở hữu chưa quyết định giấy phép. Khi import upstream, phải kiểm
tra và tuân thủ license của từng nguồn độc lập.
