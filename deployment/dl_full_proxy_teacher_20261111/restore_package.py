"""Verify <=8MiB pieces, then extract into a new directory. CPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import tempfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    root=Path(__file__).resolve().parent; m=json.loads((root/'transfer_manifest.json').read_text())
    if a.out.exists(): raise FileExistsError(a.out)
    with tempfile.TemporaryDirectory(prefix='wam_teacher_restore_') as d:
        archive=Path(d)/'payload.tar'
        with archive.open('wb') as f:
            for r in m['parts']:
                part=root/r['path']
                assert part.stat().st_size==r['bytes'] and sha(part)==r['sha256'], part
                f.write(part.read_bytes())
        assert archive.stat().st_size==m['archive_bytes'] and sha(archive)==m['archive_sha256']
        a.out.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target=(a.out/member.name).resolve()
                assert target.is_relative_to(a.out.resolve()) and member.isfile(), member.name
                with tar.extractfile(member) as source:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with target.open('wb') as dest:
                        for b in iter(lambda:source.read(8*1024**2),b''):dest.write(b)
    assert sha(a.out/'payload_manifest.json')==m['payload_manifest_sha256']
    for r in json.loads((a.out/'payload_manifest.json').read_text())['files']:
        target=a.out/r['path']; assert target.stat().st_size==r['bytes'] and sha(target)==r['sha256'], target
    print(json.dumps(dict(complete=True,output=str(a.out),archive_sha256=m['archive_sha256'])))


if __name__=='__main__': main()
