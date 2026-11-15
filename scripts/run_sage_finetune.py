"""Entry point for the PROVISIONAL SAGE Student adaptation pipeline."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.methods.sage_finetune.train import main

if __name__ == "__main__":
    main()
