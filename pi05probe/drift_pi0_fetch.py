"""CPU download of the public pi0_libero checkpoint, pinned to GCS generations."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request


def download(dest, output):
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)

    url = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi0_libero/&maxResults=1000'
    metadata = json.load(urllib.request.urlopen(url, timeout=60))
    assert not metadata.get('nextPageToken'), 'unexpected paginated checkpoint inventory'
    objects = [r for r in metadata['items'] if not r['name'].endswith('/')]
    assert objects and all(r.get('md5Hash') for r in objects), 'source checksum missing'
    save(output / 'source_inventory.json', metadata)
    scratch = dest.with_name(dest.name + '.verified-download')
    assert not dest.exists(), 'refuse to merge into an existing checkpoint'
    scratch.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    progress = dict(stage='downloading', total_files=len(objects), total_bytes=sum(int(r['size']) for r in objects),
                    completed_files=0, downloaded_bytes=0, files=[])
    last = [0.]

    def fetch(item):
        relative = item['name'].removeprefix('checkpoints/pi0_libero/')
        assert relative and '..' not in Path(relative).parts
        path = scratch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        address = 'https://storage.googleapis.com/openpi-assets/' + urllib.parse.quote(item['name'], safe='/') + '?generation=' + item['generation']
        sha, md5 = hashlib.sha256(), hashlib.md5()
        size = 0
        with urllib.request.urlopen(address, timeout=120) as response, path.open('wb') as stream:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                stream.write(block)
                sha.update(block)
                md5.update(block)
                size += len(block)
                with lock:
                    progress['downloaded_bytes'] += len(block)
                    if time.monotonic() - last[0] > 2:
                        save(output / 'progress.json', progress)
                        last[0] = time.monotonic()
        assert size == int(item['size']), ('wrong size', relative)
        assert base64.b64encode(md5.digest()).decode() == item['md5Hash'], ('source checksum mismatch', relative)
        row = dict(path=relative, bytes=size, sha256=sha.hexdigest(), md5=item['md5Hash'], generation=item['generation'])
        with lock:
            progress['completed_files'] += 1
            progress['files'].append(row)
            save(output / 'progress.json', progress)
            print('[download]', progress['completed_files'], '/', len(objects), relative, size, flush=True)
        return row

    save(output / 'progress.json', progress)
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(fetch, objects))
    assert sum(r['bytes'] for r in files) == progress['total_bytes']
    save(scratch / 'download_manifest.json', dict(source='gs://openpi-assets/checkpoints/pi0_libero',
         generation_pinned=True, source_md5_verified=True, files=files))
    scratch.rename(dest)
    save(output / 'complete.json', dict(complete=True, checkpoint=str(dest), files=len(files), bytes=progress['total_bytes'], source_md5_verified=True))
    progress['stage'] = 'complete'
    save(output / 'progress.json', progress)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    download(args.dest, args.out)
