# Chuẩn bị GitHub — Lộc bé, 15/11/2026

Branch: feature/locbe-sage-finetune-1911. Base đã fetch origin/main:
5352eedecaa2a34e703f12e917e31f03627b32b1.
[PR #8](https://github.com/quan14333/xulianh/pull/8) đã MERGED
14/11/2026, merge commit8cc907b5338efe21fd26355258cc2d05411ca875;
là ancestor của main. Không cherry-pick B3 chưa merge; không sửa source Mân.
Phạm vi lần này tạo một commit local, chưa push hoặc tạo/merge PR.

## Kiểm chứng trước commit

- **Pytest8.4.2:85 passed in8.32s**,toàn suite,không skip.
- Lần đầu75passed/10setup errors vì Windows chặn thư mục pytest-of-user cũ.
  Chạy lại với --basetemp mới đã xác minh không tồn tại; không sửa tests/code để né lỗi.
- 6entrypoint CLI --help PASS;17invocations trong tài liệu khớp flags hiện có.
- Repo validator chạy cả trong pytest và trực tiếp trước commit.
- Source/model/training recipe không đổi trong đợt Git-preparation.
- Quét file commit:chỉ UTF8 text nhỏ,không raw logs,binaries/checkpoints,dataset,
  venv/cache;không phát hiện token/private key/credential URL/home paths.
  Đây là kiểm tra pattern và allowlist,không phải chứng nhận không thể có secret.
- Thêm ignore .venv-sage,common caches,*.log và allowlist đúng4report files nhỏ.
  Các log chi tiết và SHA inventory chỉ giữ local,không stage.
- Đã cài pytest8.4.2+dev dependencies nhỏ theo requirements-dev;không đổi
  PyTorch/CUDA. Pillow13 của local vẫn ngoài repo constraint<12;ghi rõ giới hạn,
  không âm thầm thay dependency protocol.
- Unit tests chỉ dùng fixtures tổng hợp/metadata đã commit;không chạy mô hình trên
  FMB Test. Test consistency có đọc CSV metadata Test sẵn có,không đọc ảnh Test
  hoặc sử dụng Test để chọn config/checkpoint.

## Trạng thái nghiên cứu trung thực

Student-only candidate vẫn PROVISIONAL/BLOCKED_FOR_APPROVAL.
Pilot:2microbatch/4Train images/1update/**0 full epochs**.
Không có full trained baseline hoặc mIoU mới. Quân method:
WAITING_FOR_METHOD_SPEC,interface/stub only. B3 checkpoint FMB:
NOT_AVAILABLE_NOT_AUTHORIZED;loader integration PASS không phải B3 model inference.
Objective candidate khác official SAGE. Không có claim novelty/semantic improvement.

Báo cáo lịch sử audit/implementation/handoff giữ snapshot cũ,timestamp/commit cũ,
unittest34/45 và trạng thái chưa cópytest khi đo. Tài liệu này ghi bước Git-preparation
mới;không viết lại lịch sử thực nghiệm. Evidence nhỏ:
[report README](../../reports/experiments/locbe_sage_finetune/README.md).
Lệnh GPU/provenance/config:
[handoff](HANDOFF_19NOV.md),[module README](../../src/methods/sage_finetune/README.md).

## Danh sách chính xác39file trong commit

- .gitignore
- configs/methods/b3_variants.yaml
- configs/methods/sage_student_finetune.yaml
- docs/locbe_finetune/B3_INTEGRATION_PLAN.md
- docs/locbe_finetune/BASELINE_PROPOSAL.md
- docs/locbe_finetune/CONFIG_DECISIONS.md
- docs/locbe_finetune/FINETUNE_AUDIT.md
- docs/locbe_finetune/GITHUB_PREP.md
- docs/locbe_finetune/HANDOFF_19NOV.md
- docs/locbe_finetune/IMPLEMENTATION_REPORT.md
- reports/experiments/locbe_sage_finetune/FINETUNE_TRAINING_REPORT.md
- reports/experiments/locbe_sage_finetune/README.md
- reports/experiments/locbe_sage_finetune/public_evidence.json
- reports/experiments/locbe_sage_finetune/train_history.csv
- scripts/check_b3_inputs.py
- scripts/export_b3_variant.py
- scripts/export_sage_finetuned.py
- scripts/run_sage_finetune.py
- src/methods/b3_adapter.py
- src/methods/b3_loader_adapter.py
- src/methods/fusion_interface.py
- src/methods/quan_method/README.md
- src/methods/quan_method/__init__.py
- src/methods/quan_method/interface.py
- src/methods/sage_finetune/README.md
- src/methods/sage_finetune/__init__.py
- src/methods/sage_finetune/checkpoint.py
- src/methods/sage_finetune/config.py
- src/methods/sage_finetune/dataset.py
- src/methods/sage_finetune/infer.py
- src/methods/sage_finetune/losses.py
- src/methods/sage_finetune/model.py
- src/methods/sage_finetune/train.py
- tests/sage_finetune_helpers.py
- tests/test_b3_adapter.py
- tests/test_sage_finetune_checkpoint.py
- tests/test_sage_finetune_dataset.py
- tests/test_sage_finetune_inference.py
- tests/test_sage_finetune_losses.py

## Commit và PR dự kiến

Message/PR title:
~~~text
feat(methods): add SAGE Student finetuning baseline and B3 integration
~~~

PR body được chuẩn bị thành file local ngoài repository,không đăng tự động.
Reviewer cần: **Quân** (objective/config/method spec) và **Mân**
(B3 input contract,class mapping,metrics/fair protocol).
Commit SHA cuối cùng và git status được in sau commit,không chèn SHA tự tham chiếu vào tài liệu.

## Reproduce kiểm tra

Từ root repository dùng Python môi trường có torch,OpenCV,numpy,PyYAML,pandas,
transformers và pytest. Phiên bản runtime GPU được ghi trong báo cáo;
đây chưa phải lockfile portable đa nền tảng.

~~~powershell
$py = 'D:\SAGE_Research\xulianh\.venv-sage\Scripts\python.exe'
$pytestTemp = Join-Path ([IO.Path]::GetTempPath()) ('sage-pytest-' + [guid]::NewGuid().ToString('N'))
if (Test-Path -LiteralPath $pytestTemp) { throw 'Use a new pytest temp path' }
& $py -X utf8 -B -m pytest -q -p no:cacheprovider --basetemp $pytestTemp
& $py -X utf8 -B scripts/validate_repo.py
& $py -X utf8 -B scripts/export_b3_variant.py --help
& $py -X utf8 -B scripts/check_b3_inputs.py --help
~~~

Full training không nằm trong commit-preparation. Cần review objective bằng record
khớp config fingerprint,B3 checkpoint và method spec trước các bước tương ứng.
