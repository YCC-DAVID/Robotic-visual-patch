"""Verify and join small export parts; extract into a NEW destination."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.parts / 'manifest.json').read_text())
    for row in manifest['parts']:
        path = args.parts / row['name']
        if path.name != row['name'] or path.stat().st_size != row['bytes']:
            raise RuntimeError('Invalid part: ' + row['name'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('Checksum mismatch: ' + row['name'])
    args.destination.mkdir(parents=True, exist_ok=False)
    for group in ['code', 'observations']:
        archive = args.destination / (group + '.tar.gz')
        with archive.open('xb') as output:
            for row in manifest['parts']:
                if row['name'].startswith(group + '.tar.gz.part'):
                    output.write((args.parts / row['name']).read_bytes())
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path = Path(member.name)
                if path.is_absolute() or '..' in path.parts or not member.isfile():
                    raise RuntimeError('Invalid archive entry: ' + member.name)
            tar.extractall(args.destination)
    root = args.destination / 'WAMattack'
    for rows in manifest['files'].values():
        for row in rows:
            path = root / row['path']
            if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise RuntimeError('Restored file mismatch: ' + row['path'])
    print('Verified and restored:', root)


if __name__ == '__main__':
    main()
