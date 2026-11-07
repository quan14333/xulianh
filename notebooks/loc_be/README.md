# Bàn giao SAGE của Lộc bé — 08/11/2026

Ba notebook là source đã dọn output, không có credential/attachment. Chúng cần
môi trường Kaggle và dependency model ngoài môi trường CI của repo.

| Notebook | Phạm vi / trạng thái |
|---|---|
| `SAGE_LocBe_Inference_V4.ipynb` | Pretrained Student; run ID riêng, manifest/config/log; pin SAGE và checkpoint theo run gốc. **V4 chưa chạy lại trên GPU**; kết quả GPU đã có là bản v3 ngày 02/11 |
| `SAGE_08Nov_ThreeWay_Comparison.ipynb` | Đọc PNG SAGE, so Y_AVG/Y_MAX; có IR_hot_ratio top-decile, khác heuristic-mask trong notebook PCDFusion |
| `SAGE_PCDFusion_RegionAudit.ipynb` | PCDFusion pretrained, cùng 10 ID; 4 phương pháp, paired deltas, heuristic-mask IR. Audit 9 hình chữ nhật được tái lập riêng bằng script dưới đây |

Khi import, chỉ pin commit/checkpoint SAGE, sửa tên output/ZIP V4 trong hướng dẫn,
bỏ tên dataset gắn tài khoản khỏi notebook PCDFusion và chuẩn hóa JSON. Công thức
metric notebook giữ nguyên, cần Lộc lớn review trước khi dùng cho benchmark.
Chi tiết checksum/thay đổi: [source provenance](../../reports/tables/loc_be/source_provenance.json).

## Artifact ngoài Git

Người dùng cung cấp hai ZIP, không commit chúng:

- `SAGE_LocBe_pretrained_inference_results.zip`
- `sage_pcdfusion_08nov_20261106_170517.zip`

SHA256 ZIP và kết quả replay: [offline_validation.json](../../reports/tables/loc_be/offline_validation.json).
Kaggle Output/storage URL có quyền truy cập: **link pending**. Liên hệ người giữ
artifact để lấy đúng ZIP; không dùng URL tự suy đoán từ đường dẫn Kaggle Input.

Giải nén **ngoài repo**, ví dụ `C:\Temp\locbe-artifacts\`, sao cho hai thư mục có:

```text
SAGE_LocBe_pretrained_inference_results/
  fused_png/<image_id>.png
sage_pcdfusion_08nov_20261106_170517/
  dataset/FMB/test/ir/<image_id>.png
  dataset/FMB/test/vi/<image_id>.png
  fused/PCDFusion/<image_id>.png
```

IDs: `00014, 00024, 00061, 00064, 00071, 00075, 00104, 00109, 00112, 00145`.
Đây là **FMB/test repackaged 640×512**, khác audit FMB chính thức 800×600 của nhóm;
không resize để giả định hai nguồn tương đương. SAGE v3 dùng seed=2, frozen
Student, input Y-visible/IR, phục hồi màu theo upstream. Model PCDFusion frozen,
pad_multiple=32; cấu hình đầy đủ trong [run config](../../reports/tables/loc_be/run_config.json).

## Chạy lại ROI bằng CPU

Cài môi trường repo theo README gốc. OpenCV là dependency **tùy chọn cho đọc PNG**;
unit test số học không cần OpenCV, GPU, PyTorch, checkpoint hoặc tải dataset.

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install opencv-python-headless
python -m src.analysis.controlled_roi_audit --sage-root C:\Temp\locbe-artifacts\SAGE_LocBe_pretrained_inference_results --pcdfusion-root C:\Temp\locbe-artifacts\sage_pcdfusion_08nov_20261106_170517 --output-dir C:\Temp\locbe-artifacts\roi-replay-new
```

`--output-dir` phải chưa tồn tại; script từ chối ghi đè run trước. Cũng có thể gọi
`python src/analysis/controlled_roi_audit.py` với cùng ba tham số.

Output: `roi_primary.csv`, `roi_sensitivity_all.csv`, `roi_robustness_flags.csv`,
`file_provenance.csv`, `summary.json`. Ảnh nguồn không thay đổi; script không xuất
ảnh dataset vào Git. Provenance mới dùng `root_argument` và đường dẫn tương đối
trong từng root; đối chiếu hash theo `(image_id, method)` với bảng gốc.

Đối chiếu mọi ô trong ba CSV replay với bản gốc (chạy Python từ repo):

```python
import csv
from pathlib import Path

replay = Path("C:/Temp/locbe-artifacts/roi-replay-new")
reference = Path("reports/tables/loc_be")
def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
for name in ("roi_primary.csv", "roi_sensitivity_all.csv", "roi_robustness_flags.csv"):
    assert rows(replay / name) == rows(reference / name), name
```

9 tọa độ ROI và 6 cấu hình giữ nguyên bản audit: rectangle nửa mở `[x1,x2),[y1,y2)`,
ring chữ nhật được clip theo biên ảnh và loại mọi ROI khác trong cùng cảnh.
Contrast = mean/median(ROI) − mean/median(ring), float64; VIS/SAGE/PCDFusion dùng
OpenCV BGR→YCrCb lấy kênh Y. Đây là chú thích **thủ công, hậu nghiệm**, không phải
segmentation GT; mô tả người/xe chưa có nhãn xác nhận. Ngưỡng IR-salient gốc:
contrast IR ≥ 12 và contrast IR − VIS ≥ 12, giữ cố định khi replay.

## Kiểm tra source và bảng

```powershell
python -m compileall -q src scripts tests
python scripts/validate_repo.py
python -m pytest
python -m pip install nbformat
```

Kiểm tra schema notebook (không chạy inference):

```python
import ast
from pathlib import Path
import nbformat

for path in Path("notebooks/loc_be").glob("*.ipynb"):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None
```

Đợt tích hợp đã replay đủ 9/54 dòng ROI và 9 dòng robustness, khớp **1.449 ô CSV**
gốc; 40/40 hash file và 10/10 hash pixel RGB SAGE khớp. Không chạy lại GPU inference.
Xem [báo cáo kết quả và giới hạn](../../reports/weekly/loc_be_2026-11-08.md).
