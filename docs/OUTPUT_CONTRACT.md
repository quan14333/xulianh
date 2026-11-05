# Output contract

## Cấu trúc chuẩn

```text
runs/<run_id>/
├── run_config.json
├── environment.json
├── manifest.csv
├── logs/
│   └── run.log
├── fused/
│   └── <image_id>.png
├── metrics/
│   ├── per_image.csv
│   ├── summary.json
│   └── paired_comparison.csv
└── figures/
```

`run_id` phải duy nhất, gợi ý `YYYYMMDD-HHMM_<method>_<split>_<shortsha>`; không ghi đè run cũ.
`run_config.json` ghi method/config/checkpoint/seed/preprocessing/commit. `environment.json` ghi
Python, package, CUDA/GPU và platform. Manifest là snapshot input có image ID và đường dẫn tương đối.

## Checklist theo loại run

| Loại run | File bắt buộc |
|---|---|
| Smoke inference | `run_config.json`, `environment.json`, `manifest.csv`, `logs/run.log`, `fused/` |
| Baseline inference | Các file smoke + fused đủ theo manifest + trạng thái từng ảnh |
| Metrics-only | `run_config.json`, `environment.json`, `manifest.csv`, `logs/run.log`, `metrics/per_image.csv`, `metrics/summary.json` |
| Paired comparison | Các file metrics-only + `metrics/paired_comparison.csv` |
| Training/prototype | Config/environment/manifest/log + checkpoint ID/nơi lưu artifact + learning curves khi có |

Không phải run nào cũng có mọi file; ví dụ inference-only chưa cần `paired_comparison.csv`.

## Schema tối thiểu

- `per_image.csv`: `image_id`, metric columns, `valid`, `error_reason`.
- `summary.json`: run ID, metric definitions/versions, mean/std/median/valid count.
- `paired_comparison.csv`: image ID, baseline value, candidate value, delta cho từng metric.
- `environment.json`: không chứa token, username hoặc đường dẫn riêng tư không cần thiết.

## Git và provenance

Artifact lớn lưu ngoài Git (Kaggle Working Output hoặc storage nhóm). Git chỉ lưu code, config,
manifest, tài liệu, bảng nhỏ và hướng dẫn/URI lấy artifact. Mọi số liệu trong báo cáo phải truy về
một run cụ thể; nếu artifact hết hạn thì ghi trạng thái, không tự tạo lại số liệu để lấp chỗ trống.
