import sys
import torch

print(f"Python executable: {sys.executable}")
print(f"\nPython path:")
for p in sys.path:
    print(f"  {p}")
print(f"\nTorch location: {torch.__file__}")
print(f"Torch version: {torch.__version__}")
