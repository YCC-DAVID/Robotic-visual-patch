"""CPU-resident measured EOT bank, with only the active frame copied to GPU."""
from render_proxy import Pool


class Bank:
    def __init__(self, root, keys, device, seeds=None):
        self.pools = {}; self.device = device
        self.active = Pool.__new__(Pool)
        self.active.device = device; self.active.global_response = True
        self.active.blur_sigma = 0.; self.active.offset = .5
        for key in keys:
            paths = sorted((root/key).glob('seed*.npz'))
            if seeds is not None: paths = [p for p in paths if int(p.stem[4:]) in seeds]
            assert paths, (key, 'empty pool')
            self.pools[key] = Pool(paths, 'cpu', global_response=True)
        self.keys = list(keys)
        reference = [(f['seed'], f['query']) for f in self.pools[self.keys[0]].frames]
        for key, pool in self.pools.items():
            assert [(f['seed'], f['query']) for f in pool.frames] == reference, key

    def frames(self, key): return self.pools[key].frames

    def images(self, q, key, i):
        frame = self.frames(key)[i]
        self.active.frames = [{'views':{n:{k:v.to(self.device) for k,v in view.items()}
                                        for n,view in frame['views'].items()}}]
        return self.active.images(q, 0, 'proxy')
