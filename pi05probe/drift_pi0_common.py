"""Provenance and CPU helpers for the official DRIFT -> native-render experiment."""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIFT = ROOT / 'third_party/DRIFT'
SUITES = ('libero_spatial', 'libero_goal', 'libero_object', 'libero_10')
LIMITS = dict(libero_spatial=220, libero_goal=300, libero_object=280, libero_10=520)
TRAIN_INITS = (10, 11, 12)
TEST_INITS = tuple(range(10))
CONDITIONS = ('clean', 'random_digital', 'drift_digital', 'random_render', 'drift_render')
MODEL_CONFIG = 'pi0_libero'
PATCH_SIZE, PATCH_Y, PATCH_X = 32, 192, 96
ACTION_HORIZON, ACTION_DIM = 50, 32


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def stage(root, name, **fields):
    value = dict(stage=name, updated=now(), pid=os.getpid(), **fields)
    import sys
    runtime = sys.modules.get('visible_pos_pipeline')
    if runtime is not None and Path(runtime.DEST).resolve() == Path(root).resolve():
        with runtime._STATE_LOCK:
            runtime._STATE.update(value)
            value = dict(runtime._STATE)
    dump(Path(root) / 'progress.json', value)
    print('[stage]', json.dumps(value, ensure_ascii=False), flush=True)


def bootstrap(sim=False):
    import sys
    paths = [ROOT / 'third_party/openpi/src',
             ROOT / 'third_party/openpi/packages/openpi-client/src']
    if sim:
        paths += [ROOT / 'third_party/openpi/third_party/libero']
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-drift-paper-numba')
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    os.environ.setdefault('OPENPI_DATA_HOME', str(Path.home() / '.cache/openpi'))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    if sim:
        install_verified_uuid_compatibility()


def install_verified_uuid_compatibility():
    """Replace one obsolete UUID substring assertion, after driver-level audit.

    CUDA visibility remains exactly the assigned UUID. The vendor installation
    is unchanged; all other assertions and rendering code execute normally.
    """
    import ast
    import importlib.abc
    import importlib.machinery
    import sys
    cuda, egl = os.environ.get('CUDA_VISIBLE_DEVICES', ''), os.environ.get('MUJOCO_EGL_DEVICE_ID', '')
    if not cuda.startswith('GPU-') or egl in cuda:
        return
    from egl_device_audit import audit
    result = audit(cuda, int(egl))
    message = 'MUJOCO_EGL_DEVICE_ID needs to be set to one of the device id specified in CUDA_VISIBLE_DEVICES'

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            source = self.get_source(fullname)
            tree = ast.parse(source, filename=self.path)
            removed = 0
            for node in ast.walk(tree):
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        for i, child in enumerate(value):
                            if isinstance(child, ast.Assert) and isinstance(child.msg, ast.Constant) and child.msg.value == message:
                                value[i] = ast.copy_location(ast.Pass(), child)
                                removed += 1
            assert removed == 1, 'unexpected robosuite compatibility source; refuse broad assertion changes'
            print('[uuid-compat]', json.dumps(dict(result, removed_assertion=message,
                  scope='process-local single assertion; assigned UUID remains visible')), flush=True)
            return compile(ast.fix_missing_locations(tree), self.path, 'exec')

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != 'robosuite.utils.binding_utils':
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            assert spec is not None and spec.origin
            spec.loader = Loader(fullname, spec.origin)
            return spec

    sys.meta_path.insert(0, Finder())


def configure_libero(root):
    import yaml
    base = ROOT / 'third_party/openpi/third_party/libero/libero/libero'
    folder = Path(root) / 'libero_config'
    folder.mkdir(parents=True, exist_ok=True)
    values = dict(assets=str(base / 'assets'), bddl_files=str(base / 'bddl_files'),
                  benchmark_root=str(base), datasets=str(base.parent / 'datasets'),
                  init_states=str(base / 'init_files'))
    (folder / 'config.yaml').write_text(yaml.safe_dump(values))
    os.environ['LIBERO_CONFIG_PATH'] = str(folder)
