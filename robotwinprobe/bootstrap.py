"""Project-local paths; keep simulator and model Python environments separate."""
from pathlib import Path
import os
import sys

PROBE = Path(__file__).resolve().parent
ROOT = PROBE.parent
ROBOTWIN = ROOT/'third_party/RoboTwin'
OUT = PROBE/'out'


def simulator_paths():
    for path in [PROBE/'runtime_py', ROBOTWIN/'envs/curobo/src', ROBOTWIN,
                 ROBOTWIN/'scripts', ROBOTWIN/'description/utils',
                 ROOT/'third_party/openpi/packages/openpi-client/src']:
        sys.path.insert(0, str(path))
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.1')
    os.environ.setdefault('TORCH_EXTENSIONS_DIR', str(OUT/'torch_extensions'))
    os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.9')
    os.environ.setdefault('MAX_JOBS', '4')
    os.environ['PATH'] = str(PROBE/'runtime_py/bin')+os.pathsep+os.environ['PATH']


def model_paths():
    for path in [ROOT/'third_party/openpi/packages/openpi-client/src',
                 ROOT/'third_party/openpi/src', ROOT/'third_party/transformers_patched']:
        sys.path.insert(0, str(path))
    os.environ.setdefault('OPENPI_DATA_HOME', '/home/user1/.cache/openpi')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
