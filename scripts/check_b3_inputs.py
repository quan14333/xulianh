"""Validate PNGs through the original B3 segmentation Dataset on selected Dev IDs."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.methods.b3_loader_adapter import main

if __name__ == "__main__":
    main()
