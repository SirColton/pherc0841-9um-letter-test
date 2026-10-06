# Reader A view maker: crops and downsamples ONLY ScrollPrize's 2.4 um ink map of PHerc0841 252405.
# Usage: day1-refread-views.py overview <ds>
#        day1-refread-views.py crop <r0> <c0> <h> <w> <ds> <name> [--mirror] [--grid STEP]
import sys, os, numpy as np, tifffile
from PIL import Image, ImageDraw, ImageFont
MAP = 'geom/day1-9um/render/src/inkmap.tif'
OUT = 'geom/day1-9um/reference/views'
os.makedirs(OUT, exist_ok=True)
a = tifffile.imread(MAP)
H, W = a.shape

def pool(x, ds):
    h, w = (x.shape[0] // ds) * ds, (x.shape[1] // ds) * ds
    return x[:h, :w].reshape(h // ds, ds, w // ds, ds).astype(np.float32).mean((1, 3))

def save(img, name, r0, c0, ds, grid):
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert('RGB')
    if grid:
        d = ImageDraw.Draw(im)
        g0r = ((r0 + grid - 1) // grid) * grid
        g0c = ((c0 + grid - 1) // grid) * grid
        for gr in range(g0r, r0 + img.shape[0] * ds, grid):
            y = (gr - r0) // ds
            d.line([(0, y), (im.width, y)], fill=(255, 140, 0), width=1)
            d.text((2, y + 1), str(gr), fill=(255, 140, 0))
        for gc in range(g0c, c0 + img.shape[1] * ds, grid):
            x = (gc - c0) // ds
            d.line([(x, 0), (x, im.height)], fill=(0, 160, 255), width=1)
            d.text((x + 2, 2), str(gc), fill=(0, 160, 255))
    p = os.path.join(OUT, name + '.png')
    im.save(p)
    print(p, im.size)

if sys.argv[1] == 'overview':
    ds = int(sys.argv[2])
    grid = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    save(pool(a, ds), 'overview-ds%d' % ds, 0, 0, ds, grid)
    print('map', H, W)
else:
    r0, c0, h, w, ds = map(int, sys.argv[2:7]); name = sys.argv[7]
    grid = 0
    if '--grid' in sys.argv: grid = int(sys.argv[sys.argv.index('--grid') + 1])
    sub = a[max(0, r0):min(H, r0 + h), max(0, c0):min(W, c0 + w)]
    img = pool(sub, ds)
    if '--stretch' in sys.argv:
        lo, hi = np.percentile(img, [1, 99.5]); img = (img - lo) / max(1e-6, hi - lo) * 255
    if '--mirror' in sys.argv:
        img = img[:, ::-1]; grid = 0
    save(img, name, r0, c0, ds, grid)
