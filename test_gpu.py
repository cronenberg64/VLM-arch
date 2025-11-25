import torch
import sys

print(f'Python executable: {sys.executable}')
print(f'PyTorch location: {torch.__file__}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
else:
    print(f'CUDA version: {torch.version.cuda}')
    print('GPU Device: None')
