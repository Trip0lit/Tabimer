import numpy as np, cv2, sys, json
from scipy import ndimage as ndi
OUT='../assets/'
img=cv2.imread('../assets/source.webp'); H,W=img.shape[:2]
d=np.load('disp.npy').astype(np.float32)
# sharpen depth edges: snap to local near/far value
mx=ndi.grey_dilation(d,size=7); mn=ndi.grey_erosion(d,size=7)
edge=(mx-mn)>0.35
snap=np.where(np.abs(d-mx)<np.abs(d-mn),mx,mn)
d=np.where(edge,snap,d)
for _ in range(3):
    d=cv2.bilateralFilter(d,15,0.25,9)
# straighten the columns: a column is vertical, so it keeps one depth from capital to base
COLS=[(35,95,0,600,3.65,0.5),(230,340,0,560,1.67,0.45),(705,752,0,560,1.75,0.4),(755,845,0,610,2.7,0.6)]
colmask=np.zeros(d.shape,bool)
for x0,x1,y0,y1,dc,tol in COLS:
    sub=d[y0:y1,x0:x1]; m=np.abs(sub-dc)<tol
    m=ndi.binary_opening(m,np.ones((5,3)))
    sub[m]=dc; colmask[y0:y1,x0:x1]|=m
# background disparity: remove near objects (figures, columns, plant)
bg=ndi.grey_erosion(d,size=(5,71))
bg=ndi.grey_erosion(bg,size=(15,1))
bg=cv2.GaussianBlur(bg,(0,0),6)
bg=np.minimum(bg,d)
diff=d-bg
mask=(diff>0.3).astype(np.uint8)
mask=cv2.dilate(mask,np.ones((11,11),np.uint8))
cv2.imwrite('mask.png',mask*255)
# texture-preserving fill: mirror the pattern from both sides of each masked run, cross-faded
def mirror_fill(img,mask):
    out=img.astype(np.float32).copy(); H,W=mask.shape; filled=np.zeros_like(mask)
    def refl(row,start,step,n,lo,hi):
        L=hi-lo
        if L<=0: return None
        idx=np.arange(n); p=idx%(2*L); p=np.where(p<L,p,2*L-1-p)
        return row[(start-1-p) if step<0 else (start+p)] if True else None
    for y in range(H):
        m=mask[y]; x=0
        while x<W:
            if not m[x]: x+=1; continue
            a=x
            while x<W and m[x]: x+=1
            b=x; n=b-a
            # free gaps on each side
            la=a
            while la>0 and not m[la-1]: la-=1
            rb=b
            while rb<W and not m[rb]: rb+=1
            row=img[y].astype(np.float32)
            Lw=a-la; Rw=rb-b
            left=right=None
            if Lw>0:
                i=np.arange(n); p=i%(2*Lw); p=np.where(p<Lw,p,2*Lw-1-p); left=row[a-1-p]
            if Rw>0:
                i=np.arange(n)[::-1]; p=i%(2*Rw); p=np.where(p<Rw,p,2*Rw-1-p); right=row[b+p]
            if left is None and right is None: continue
            if left is None: v=right
            elif right is None: v=left
            else:
                w=np.clip((np.arange(n)+0.5)/n,0,1)[:,None]; w=w*w*(3-2*w); v=left*(1-w)+right*w
            out[y,a:b]=v; filled[y,a:b]=1
    return out.clip(0,255).astype(np.uint8),filled
fill,filled=mirror_fill(img,mask)
fb=cv2.GaussianBlur(fill,(0,0),sigmaX=0.6,sigmaY=2.2)
bgimg=np.where(mask[...,None]>0,fb,img)
rest=(mask>0)&(filled==0)
if rest.any(): bgimg=cv2.inpaint(bgimg,rest.astype(np.uint8),5,cv2.INPAINT_TELEA)
# disparity -> metric depth (metres). near floor ~2.4m, back alcove ~16m
a=0.0703; b=0.0203
z=1/(a*d+b); zb=1/(a*bg+b)*1.015
# round the columns (cylinder bulge toward the viewer)
TX=np.tan(np.radians(21))*W/H
lab,nl=ndi.label(colmask,structure=[[0,0,0],[1,1,1],[0,0,0]])
for sl in ndi.find_objects(lab):
    y=sl[0].start; x0,x1=sl[1].start,sl[1].stop
    if x1-x0<4: continue
    xs=np.arange(x0,x1); t=(xs-(x0+x1-1)/2)/((x1-x0)/2)
    rz=(x1-x0)/2*z[y,x0:x1]*2*TX/W
    z[y,x0:x1]-=rz*np.sqrt(np.clip(1-t*t,0,1))
# upscale painting x2 with gentle sharpening
def up(im):
    u=cv2.resize(im,(W*2,H*2),interpolation=cv2.INTER_LANCZOS4)
    bl=cv2.GaussianBlur(u,(0,0),1.6)
    return cv2.addWeighted(u,1.45,bl,-0.45,0)
cv2.imwrite(OUT+'painting.jpg',up(img),[cv2.IMWRITE_JPEG_QUALITY,92])
cv2.imwrite(OUT+'background.jpg',up(bgimg),[cv2.IMWRITE_JPEG_QUALITY,88])
GW,GH=600,420
def g(a): return cv2.resize(a,(GW,GH),interpolation=cv2.INTER_NEAREST if a is z else cv2.INTER_AREA)
zf=cv2.resize(z,(GW,GH),interpolation=cv2.INTER_NEAREST); zbf=cv2.resize(zb,(GW,GH),interpolation=cv2.INTER_AREA)
buf=np.concatenate([np.clip(zf*1000,0,65535).astype('<u2').ravel(),np.clip(zbf*1000,0,65535).astype('<u2').ravel()])
v=buf.reshape(2*GH,GW)   # foreground rows, then background rows; R=high byte, G=low byte
png=np.zeros((2*GH,GW,3),np.uint8); png[...,2]=v>>8; png[...,1]=v&255
cv2.imwrite(OUT+'depth.png',png)
json.dump({'gw':GW,'gh':GH,'zmin':float(z.min()),'zmax':float(z.max())},open(OUT+'meta.json','w'))
print(z.min(),z.max(),zb.min(),zb.max(),mask.mean())
cv2.imwrite('bgprev.jpg',bgimg)
