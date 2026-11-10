"""Rebuild the exact source safetensors using one repaired tensor and local weights."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import struct


def copy_bytes(src, dst, count, digest):
    while count:
        data = src.read(min(count, 8 * 1024 * 1024))
        if not data:
            raise RuntimeError('Truncated input')
        dst.write(data)
        digest.update(data)
        count -= len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', type=Path)
    parser.add_argument('local_weights', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    m = json.loads((args.parts / 'manifest.json').read_text())
    for row in m['parts']:
        if Path(row['name']).name != row['name']:
            raise RuntimeError('Invalid part name')
        part = args.parts / row['name']
        data = part.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise RuntimeError('Part checksum mismatch: ' + row['name'])
    source_prefix = base64.b64decode(m['source_header_base64'])
    source_header = json.loads(source_prefix[8:])
    with args.local_weights.open('rb') as local:
        size = struct.unpack('<Q', local.read(8))[0]
        local_header = json.loads(local.read(size))
        with args.output.open('xb') as output:
            output.write(source_prefix)
            for key, value in source_header.items():
                if key == '__metadata__':
                    continue
                a, b = value['data_offsets']
                output.seek(len(source_prefix) + a)
                digest = hashlib.sha256()
                if key == m['tensor']:
                    count = 0
                    for row in m['parts']:
                        with (args.parts / row['name']).open('rb') as part:
                            copy_bytes(part, output, row['bytes'], digest)
                        count += row['bytes']
                    if count != b - a or digest.hexdigest() != m['tensor_sha256']:
                        raise RuntimeError('Tensor checksum mismatch')
                else:
                    existing = local_header[key]
                    x, y = existing['data_offsets']
                    if existing['shape'] != value['shape'] or existing['dtype'] != value['dtype'] or y-x != b-a:
                        raise RuntimeError('Tensor structure mismatch: ' + key)
                    local.seek(8 + size + x)
                    copy_bytes(local, output, b-a, digest)
    digest = hashlib.sha256()
    with args.output.open('rb') as output:
        for data in iter(lambda: output.read(8 * 1024 * 1024), b''):
            digest.update(data)
    if args.output.stat().st_size != m['source_file_bytes'] or digest.hexdigest() != m['source_file_sha256']:
        raise RuntimeError('Rebuilt full file differs from source; keep original unchanged')
    print('Exact source weights verified:', args.output, digest.hexdigest())


if __name__ == '__main__':
    main()
