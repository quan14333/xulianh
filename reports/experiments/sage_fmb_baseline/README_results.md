# Báo cáo SAGE Student baseline — Quân

- Run ID: `sage_fmb_test_student_20261106T160350Z`
- Dataset: `/kaggle/input/datasets/quannh206/data-fmbtest/test` (split **test**, FULL SPLIT)
- FMB pairs: 280; đã chạy: 280; ảnh fused hợp lệ: 280
- Source commit: `377fdbef6b8b6ccbd1139358ad66bb41ea084dc1`
- Checkpoint SHA256: `ecfde477ca4cd65401f207d5dc91ce6c475a49891d768dbdb7a66366255ee836`
- GPU: Tesla T4
- Student parameters: 135,617
- Forward mean / median / p95 (ms): 74.625 / 75.409 / 75.467
- Peak CUDA allocated (MiB): 510.589
- QA: 280/280 PASS

**Phạm vi kết luận:** Pretrained Student inference thành công trên 280 ảnh test FMB. Không suy ra đã train lại Teacher/Student. Không khẳng định tái lập 61.2% FMB mIoU, vì còn cần downstream segmentation evaluation. Chưa có bảng EN/SD/SCD/MS-SSIM nếu chưa chạy riêng evaluation pipeline.

**Files:** manifest.csv, manifest_all.csv, inference.csv, qa_per_image.csv, qa_summary.json, run_config.json, environment.json, pip_freeze.txt, preview_ir_vis_fused.png, fused/*.png.
