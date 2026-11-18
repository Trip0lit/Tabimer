"""Room model ("tour into the picture") for the painting.

Painter camera at the origin, looking along +z (three.js: -Z). One-point perspective:
principal point (CX, CY) = vanishing point, focal F px, eye 1.55 m above the floor.
Each pixel is assigned either to the modelled room (floor, walls, ceilings, alcove,
gallery) or to an object (figures, furniture, columns...) kept in relief in front.
"""
import numpy as np, cv2, json
from scipy import ndimage as ndi

W, H = 1000, 699
F, CX, CY = 930.0, 500.0, 380.0
EYE = 1.55
ROOM = dict(floor=-EYE, back=13.6, alcove=16.3, side=7.3, portico_x=2.9,
            ceil_side=3.43, ceil_gallery=3.15, gallery_z=10.5, ceil_center=5.2, front=-1.5)
CAL = (0.0261868, 0.05255132)          # 1/z = a*disp + b, fitted on the floor
OUT = '../assets/'

img = cv2.imread(OUT + 'source.webp')
d = np.load('disp.npy').astype(np.float32)

# --- depth cleanup (as before): crisp edges, smooth inside objects, straight columns
mx = ndi.grey_dilation(d, size=7); mn = ndi.grey_erosion(d, size=7)
d = np.where((mx - mn) > 0.35, np.where(np.abs(d - mx) < np.abs(d - mn), mx, mn), d)
for _ in range(3): d = cv2.bilateralFilter(d, 15, 0.25, 9)
COLS = [(35, 95, 0, 600, 3.65, 0.5), (230, 340, 0, 560, 1.67, 0.45), (705, 752, 0, 560, 1.75, 0.4), (755, 845, 0, 610, 2.7, 0.6)]
colmask = np.zeros(d.shape, bool)
for x0, x1, y0, y1, dc, tol in COLS:
    sub = d[y0:y1, x0:x1]; m = ndi.binary_opening(np.abs(sub - dc) < tol, np.ones((5, 3)))
    sub[m] = dc; colmask[y0:y1, x0:x1] |= m
# below the capitals keep only the shafts (figures stand right next to them)
for (x0, x1, ys, sx0, sx1) in [(705, 752, 300, 719, 752), (230, 340, 295, 259, 297), (755, 845, 240, 780, 820), (35, 95, 235, 50, 95)]:
    colmask[ys:, x0:sx0] = False; colmask[ys:, sx1:x1] = False
zcal = 1 / np.maximum(CAL[0] * d + CAL[1], 1 / 40)

# --- ray-cast the room model
xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
rx, ry = (xs - CX) / F, -(ys - CY) / F
INF = np.full((H, W), np.inf, np.float32)
def hitY(Y):
    with np.errstate(divide='ignore'): z = Y / ry
    return np.where(z > 0, z, np.inf)
R = ROOM
zf = hitY(R['floor'])
zcs = hitY(R['ceil_side']); zcs = np.where(np.abs(rx * zcs) > R['portico_x'], zcs, np.inf)
zcg = hitY(R['ceil_gallery']); zcg = np.where((np.abs(rx * zcg) <= R['portico_x']) & (zcg >= R['gallery_z']), zcg, np.inf)
yb = ry * R['gallery_z']
zgf = np.where((np.abs(rx * R['gallery_z']) <= R['portico_x']) & (yb >= R['ceil_gallery']), R['gallery_z'], np.inf)
# alcove openings: columns of pixels behind the arches where the painting is clearly deeper than the wall
deep = (zcal > 15.0)
top = np.full(W, -1)
for x in range(385, 705):
    c = np.where(deep[225:420, x])[0]
    if len(c): top[x] = 225 + c[0]
tv = top.astype(float); tv[tv < 0] = np.nan
tv = ndi.median_filter(np.nan_to_num(tv, nan=9999), size=25); tv[tv > 2000] = -1
yfloor_alc = CY + F * EYE / R['alcove']
alc = np.zeros((H, W), bool)
for x in range(W):
    if tv[x] > 0: alc[int(tv[x]):int(yfloor_alc) + 1, x] = True
zbk = np.where(alc, R['alcove'], R['back'])
zroom = np.minimum.reduce([zf, zcs, zcg, zgf, zbk])
zroom = np.minimum(zroom, np.abs(R['side'] / np.where(np.abs(rx) > 1e-6, rx, 1e-6)))

# --- what is room and what is an object standing in front of it
near = zcal < zroom * 0.88
obj = ndi.binary_opening(near, np.ones((3, 3)))
lab, n = ndi.label(obj); sizes = ndi.sum(obj, lab, range(1, n + 1))
obj = np.isin(lab, 1 + np.where(sizes >= 200)[0])
obj = ndi.binary_closing(obj, np.ones((5, 5))) | colmask
zbg = zroom
# --- split the objects into cards (one per figure / column / piece of furniture)
# cards reach a little below their outline onto the floor, to cover the seam with painted lower bodies
down = np.zeros((17, 3), bool); down[8:, 1] = True
ext = obj | (ndi.binary_dilation(obj, down) & (zroom == zf))
gx = np.abs(np.diff(zcal, axis=1, append=zcal[:, -1:])) / zcal
gy = np.abs(np.diff(zcal, axis=0, append=zcal[-1:, :])) / zcal
edge = (gx > 0.04) | (gy > 0.04)
core = obj & ~ndi.binary_dilation(edge, np.ones((3, 3))) & ~ndi.binary_dilation(colmask, np.ones((1, 7)))
lab, n = ndi.label(core)
for x0, x1, y0, y1, dc, tol in COLS:                  # every column is its own card
    n += 1; m = np.zeros_like(colmask); m[y0:y1, x0:x1] = colmask[y0:y1, x0:x1]; lab[m] = n
sizes = ndi.sum(lab > 0, lab, range(1, n + 1))
keep = np.zeros(n + 1, bool); keep[1:] = sizes >= 250
lab[~keep[lab]] = 0
_, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
lab = np.where(ext, lab[iy, ix], 0)
ids = np.unique(lab[lab > 0]); remap = np.zeros(lab.max() + 1, int); remap[ids] = np.arange(1, len(ids) + 1)
lab = remap[lab]
cards = []
zfg = np.zeros_like(zcal)
for k in range(1, len(ids) + 1):
    m = lab == k
    zc = float(np.median(zcal[m & obj])) if (m & obj).any() else float(np.median(zcal[m]))
    # a vertical card at depth zc, except where it would sink behind the room (then it hugs the floor/wall)
    zfg[m] = np.minimum(zc, zbg[m] * 0.995)
    xc = float(np.median((xs[m] - CX) / F * zc))
    cards.append(dict(z=round(zc, 3), x=round(xc, 3)))
assert len(cards) < 255
# --- clean room texture: objects removed, pattern continued by mirroring along each row
def mirror_fill(img, mask):
    out = img.astype(np.float32).copy(); done = np.zeros_like(mask)
    for y in range(mask.shape[0]):
        m = mask[y]; row = img[y].astype(np.float32); x = 0
        Wd = len(m)
        while x < Wd:
            if not m[x]: x += 1; continue
            a = x
            while x < Wd and m[x]: x += 1
            b = x; n = b - a; la = a
            while la > 0 and not m[la - 1]: la -= 1
            rb = b
            while rb < Wd and not m[rb]: rb += 1
            L, Rw = a - la, rb - b; left = right = None
            if L > 0: p = np.arange(n) % (2 * L); p = np.where(p < L, p, 2 * L - 1 - p); left = row[a - 1 - p]
            if Rw > 0: p = np.arange(n)[::-1] % (2 * Rw); p = np.where(p < Rw, p, 2 * Rw - 1 - p); right = row[b + p]
            if left is None and right is None: continue
            if left is None: v = right
            elif right is None: v = left
            else:
                w = ((np.arange(n) + 0.5) / n)[:, None]; w = w * w * (3 - 2 * w); v = left * (1 - w) + right * w
            out[y, a:b] = v; done[y, a:b] = 1
    return out.clip(0, 255).astype(np.uint8), done
hole = cv2.dilate(obj.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
fill, done = mirror_fill(img, hole)
fill = cv2.GaussianBlur(fill, (0, 0), sigmaX=0.6, sigmaY=2.0)
room = np.where(hole[..., None], fill, img)
# inside the alcove, continue the zigzag tiles downwards (vertical mirror) instead of sideways
ah = hole & alc
for x in np.where(ah.any(axis=0))[0]:
    col = ah[:, x]; y = 0
    while y < H:
        if not col[y]: y += 1; continue
        a0 = y
        while y < H and col[y]: y += 1
        top = a0
        while top > 0 and not col[top - 1] and alc[top - 1, x] and a0 - top < 70: top -= 1
        L = a0 - top
        if L < 4: continue
        i = np.arange(y - a0); p = i % (2 * L); p = np.where(p < L, p, 2 * L - 1 - p)
        room[a0:y, x] = room[a0 - 1 - p, x]
rest = hole & (done == 0)
if rest.any(): room = cv2.inpaint(room, rest.astype(np.uint8), 5, cv2.INPAINT_TELEA)

# --- export
def up(im):
    u = cv2.resize(im, (W * 2, H * 2), interpolation=cv2.INTER_LANCZOS4)
    return cv2.addWeighted(u, 1.45, cv2.GaussianBlur(u, (0, 0), 1.6), -0.45, 0)
cv2.imwrite(OUT + 'painting.jpg', up(img), [cv2.IMWRITE_JPEG_QUALITY, 92])
cv2.imwrite(OUT + 'room.jpg', up(room), [cv2.IMWRITE_JPEG_QUALITY, 90])
GW, GH = 600, 420
rs = lambda a: cv2.resize(a.astype(np.float32), (GW, GH), interpolation=cv2.INTER_NEAREST)
v = np.concatenate([np.clip(rs(zfg) * 1000, 0, 65535), np.clip(rs(zbg) * 1000, 0, 65535)]).astype(np.uint32)
png = np.zeros((2 * GH, GW, 3), np.uint8); png[..., 2] = v >> 8; png[..., 1] = v & 255
png[:GH, :, 0] = rs(lab).astype(np.uint8)          # blue channel: card id
cv2.imwrite(OUT + 'depth.png', png)
idm = np.zeros((H, W, 3), np.uint8); idm[..., 2] = lab        # full-resolution card mask (red = card id)
cv2.imwrite(OUT + 'cards.png', idm)
cols = []
for x0, x1, y0, y1, dc, tol in COLS:
    z = 1 / (CAL[0] * dc + CAL[1]); xc = (x0 + x1) / 2
    xs_ = np.where(colmask[300, x0:x1])[0]
    if len(xs_): xc = x0 + xs_.mean(); r = len(xs_) / 2 * z / F
    else: r = 0.13
    cols.append(dict(x=round(float((xc - CX) / F * z), 3), z=round(float(z), 3), r=round(float(r), 3)))
xa = np.where(tv > 0)[0]
ROOM.update(alcove_x0=round(float((xa.min() - CX) / F * R['alcove']), 3), alcove_x1=round(float((xa.max() + 1 - CX) / F * R['alcove']), 3),
            alcove_top=round(float((CY - tv[xa].min()) / F * R['alcove']), 3))
json.dump(dict(gw=GW, gh=GH, w=W, h=H, f=F, cx=CX, cy=CY, eye=EYE, room=ROOM, columns=cols, cards=cards), open(OUT + 'meta.json', 'w'), indent=1)
cv2.imwrite('dbg_obj.png', obj.astype(np.uint8) * 255)
cv2.imwrite('dbg_lab.png', cv2.applyColorMap((lab * 37 % 256).astype(np.uint8), cv2.COLORMAP_HSV) * (lab[..., None] > 0))
cv2.imwrite('dbg_room.jpg', room)
print('objects %.1f%%' % (obj.mean() * 100), len(cards), 'cards', ROOM)
