"""Fetch the CloudSEN12+ expert-reviewed subset, and pin what was fetched.

A survey result is only reproducible if you can say which bytes produced it. Neither
of the two earlier surveys records the version of the data it read, so neither can be
re-derived if the upstream copy is revised. This one records the resolved repository
revision and the sha256 of the archive, and writes them beside the imagery so the
survey can copy them into its own output.

CloudSEN12+ is CC0. The subset is `fixed/high.zip`: 343 entries, plain GeoTIFF,
512x512, 13 spectral bands plus an expert cloud mask.
"""
import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO = 'isp-uv-es/CloudSEN12Plus'
FILE = 'fixed/high.zip'
DEST = Path('data/cloudsen12')


def revision():
    """The commit the download resolves to, so the survey can name its input."""
    u = 'https://huggingface.co/api/datasets/{}'.format(REPO)
    return json.load(urllib.request.urlopen(u, timeout=120)).get('sha')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    zp = DEST / 'high.zip'
    url = 'https://huggingface.co/datasets/{}/resolve/main/{}'.format(REPO, FILE)

    if not zp.exists() or zp.stat().st_size == 0:
        print('downloading {} ...'.format(url), flush=True)
        urllib.request.urlretrieve(url, zp)
    print('archive {:.2f} GB'.format(zp.stat().st_size / 1e9), flush=True)

    h = hashlib.sha256()
    with open(zp, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    digest = h.hexdigest()
    print('sha256', digest, flush=True)

    out = DEST / 'high'
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.tif')]
        print('{} GeoTIFFs in archive'.format(len(names)), flush=True)
        missing = [n for n in names if not (out / Path(n).name).exists()]
        if missing:
            out.mkdir(parents=True, exist_ok=True)
            for i, n in enumerate(missing, 1):
                with z.open(n) as src, open(out / Path(n).name, 'wb') as dst:
                    dst.write(src.read())
                if i % 50 == 0:
                    print('  extracted {}/{}'.format(i, len(missing)), flush=True)

    got = sorted(out.glob('*.tif'))
    prov = {'repo': REPO, 'file': FILE, 'revision': revision(),
            'sha256': digest, 'n_tif': len(got),
            'bytes': zp.stat().st_size, 'licence': 'CC0'}
    (DEST / 'provenance.json').write_text(json.dumps(prov, indent=1))
    print('extracted {} tif to {}'.format(len(got), out))
    print('provenance ->', DEST / 'provenance.json')
    if len(got) != len(names):
        print('MISMATCH: {} in archive, {} on disk'.format(len(names), len(got)))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
