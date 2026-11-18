import onnxruntime as ort, numpy as np, sys
from PIL import Image
s=ort.InferenceSession(sys.argv[1])
im=Image.open(sys.argv[2]).convert('RGB'); W,H=im.size
mean=np.array([0.485,0.456,0.406]);std=np.array([0.229,0.224,0.225])
def run(img):
    a=np.asarray(img.resize((518,518),Image.BICUBIC),dtype=np.float32)/255
    a=((a-mean)/std).transpose(2,0,1)[None].astype(np.float32)
    return s.run(None,{'l_x_':a})[0][0]
d=(run(im)+run(im.transpose(Image.FLIP_LEFT_RIGHT))[:, ::-1])/2
d=np.asarray(Image.fromarray(d.astype(np.float32)).resize((W,H),Image.BICUBIC))
np.save('disp.npy',d)
print(d.min(),d.max())
n=(d-d.min())/(d.max()-d.min())
Image.fromarray((n*255).astype(np.uint8)).save('disp.png')
