"""CPU final verification after the experiment controller exits (kernel pidfd event)."""
import argparse
import os
from pathlib import Path
import select

from drift_pi0_common import dump, now, read, sha
from drift_pi0_validate import validate


def process_fd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Some conda Python builds omit the wrapper despite kernel support.
    import ctypes
    assert os.uname().machine in ('x86_64', 'aarch64')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(fd)


def main(root, pid):
    metadata = root / 'finalization.json'
    provenance = dict(pid=os.getpid(), controller_pid=pid, started=now(), gpu_compute=False,
        validator_sha256=sha(Path(__file__).with_name('drift_pi0_validate.py')),
        protocol_sha256=sha(root / 'protocol.json'))
    try:
        try:
            handle = process_fd(pid)
        except ProcessLookupError:
            handle = None
        if handle is not None:
            command = Path('/proc/%d/cmdline' % pid).read_bytes().split(b'\0')
            assert str(root).encode() in command and any(b'drift_pi0_pipeline.py' in c for c in command), 'controller identity mismatch'
            dump(metadata, dict(provenance, stage='waiting_for_controller',
                                wait='kernel process-exit event; no polling, signals or GPU use'))
            event = select.poll()
            event.register(handle, select.POLLIN)
            event.poll()
            os.close(handle)
        result = validate(root)
        dump(metadata, dict(provenance, stage='complete', finished=now(), validation=result,
                            summary_sha256=sha(root / 'summary.json')))
    except BaseException as error:
        dump(metadata, dict(provenance, stage='failed', finished=now(), error=repr(error),
                            note='Incomplete or invalid episodes never count as a final ASR'))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--controller-pid', type=int, required=True)
    args = p.parse_args()
    main(args.root.resolve(), args.controller_pid)
