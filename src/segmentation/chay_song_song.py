"""Train + chấm nhiều biến thể, mỗi GPU một việc cùng lúc (Kaggle T4 x2 = 2 việc song song).

Việc nào đã có kết quả thì bỏ qua. Việc bị ngắt giữa chừng thì train tiếp từ resume.pt.
Log từng việc: {RUNS}/log_{tên run}.txt

Ví dụ: python chay_song_song.py --bien_the sage vis --seed 0 --gop_dev
"""
import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
import time

import torch

from chung import DAY, RUNS
from train_seg import ten_run


def cuoi(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            ds = [d.strip() for d in f.read().splitlines() if d.strip()]
        return ds[-1][-150:] if ds else ""
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bien_the", nargs="+", required=True)
    ap.add_argument("--seed", nargs="+", type=int, default=[0])
    ap.add_argument("--model", default="b3")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tich_luy", type=int, default=1)
    ap.add_argument("--gop_dev", action="store_true")
    ap.add_argument("--lop0", action="store_true")
    ap.add_argument("--bao_moi", type=int, default=600, help="in tiến độ mỗi N giây")
    a = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    n_gpu = max(1, torch.cuda.device_count())
    viec = queue.Queue()
    ten = []
    for bt, s in itertools.product(a.bien_the, a.seed):
        t = ten_run(bt, a.model, s, a.gop_dev, a.lop0)
        if (RUNS / t / "test_ket_qua_last.json").exists():
            print(f"{t}: đã có kết quả, bỏ qua", flush=True)
            continue
        viec.put((bt, s, t))
        ten.append(t)
    print(f"{len(ten)} việc trên {n_gpu} GPU: {ten}", flush=True)
    loi = []

    def tho(gpu):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONIOENCODING": "utf-8"}
        while True:
            try:
                bt, s, t = viec.get_nowait()
            except queue.Empty:
                return
            chung = ["--model", a.model, "--epochs", str(a.epochs), "--batch", str(a.batch),
                     "--tich_luy", str(a.tich_luy), "--seed", str(s)] + (["--gop_dev"] if a.gop_dev else []) + (["--lop0"] if a.lop0 else [])
            with open(RUNS / f"log_{t}.txt", "a", encoding="utf-8") as f:
                for lenh in ([sys.executable, "train_seg.py", "--bien_the", bt, "--tiep"] + chung,
                             [sys.executable, "eval_seg.py", "--run", str(RUNS / t)]):
                    if subprocess.run(lenh, cwd=DAY, env=env, stdout=f, stderr=subprocess.STDOUT).returncode:
                        loi.append(t)
                        print(f"[GPU {gpu}] {t}: LỖI, xem {RUNS / f'log_{t}.txt'}", flush=True)
                        break
                else:
                    print(f"[GPU {gpu}] {t}: xong", flush=True)

    ds = [threading.Thread(target=tho, args=(i,)) for i in range(n_gpu)]
    for t in ds:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in ds):
        time.sleep(5)
        if time.time() - t0 >= a.bao_moi:
            t0 = time.time()
            for t in ten:
                print(f"  {t}: {cuoi(RUNS / f'log_{t}.txt')}", flush=True)
    if loi:
        for t in loi:
            print(f"---- cuối log {t} ----")
            with open(RUNS / f"log_{t}.txt", encoding="utf-8", errors="replace") as f:
                print("".join(f.readlines()[-25:]))
        sys.exit(1)


if __name__ == "__main__":
    main()
