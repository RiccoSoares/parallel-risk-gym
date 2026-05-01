import sys
import os

print('sys.executable:', sys.executable)
print('sys.version:', sys.version)
print('cwd:', os.getcwd())
print('PYTHONPATH:', os.environ.get('PYTHONPATH'))

try:
    import torch
    print('torch:', torch.__file__)
    print('torch.__version__:', torch.__version__)
    print('cuda_available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device_name:', torch.cuda.get_device_name(0))
except Exception as e:
    print('torch import failed:', type(e).__name__, e)

try:
    import ray
    print('ray:', ray.__file__)
except Exception as e:
    print('ray import failed:', type(e).__name__, e)
