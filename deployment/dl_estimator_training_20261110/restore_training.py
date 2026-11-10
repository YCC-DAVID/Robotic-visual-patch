"""Verify parts, join, extract to a new directory, and verify restored files."""
import argparse, hashlib, json, subprocess, tarfile
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

a=argparse.ArgumentParser();a.add_argument('parts',type=Path);a.add_argument('destination',type=Path);a=a.parse_args()
m=json.loads((a.parts/'parts_manifest.json').read_text())
for r in m['parts']:
    if Path(r['name']).name!=r['name']:raise ValueError('Invalid path')
    p=a.parts/r['name']
    assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],p
# Fail before extraction if destination already exists.
a.destination.mkdir(parents=True,exist_ok=False)
archive=a.destination/'estimator_training.tar.zst'
with archive.open('xb') as f:
    for r in m['parts']:
        with (a.parts/r['name']).open('rb') as p:
            for b in iter(lambda:p.read(8*1024*1024),b''):f.write(b)
assert archive.stat().st_size==m['archive_bytes'] and digest(archive)==m['archive_sha256']
proc=subprocess.Popen(['zstd','-d','-c',str(archive)],stdout=subprocess.PIPE)
try:
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for member in tar:
            if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('Invalid archive member')
            dest=a.destination/member.name;dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as out:
                stream=tar.extractfile(member)
                for b in iter(lambda:stream.read(8*1024*1024),b''):out.write(b)
    proc.stdout.close()
    assert proc.wait()==0
finally:
    proc.stdout.close()
manifest=json.loads((a.parts/'payload_manifest.json').read_text())
for r in manifest['files']:
    p=a.destination/'WAMattack'/r['path']
    assert p.stat().st_size==r['bytes'] and digest(p)==r['sha256'],p
print('Verified estimator training payload:',a.destination/'WAMattack')
