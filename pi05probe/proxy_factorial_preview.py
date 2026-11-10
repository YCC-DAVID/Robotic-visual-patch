"""Scientific contact sheet for the fixed random-texture control."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from proxy_factorial_proxy import WorldTextureProxy, VARIANTS


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent/'out/proxy_factorial'
    data=np.load(root/'calibration.npz');k=155
    q=torch.from_numpy(data['random__texture'].astype(np.float32).transpose(2,0,1)/255)[None]
    images=[data['clean_base'][k],np.asarray(Image.fromarray(data['random256'][k]).resize((224,224),Image.Resampling.BILINEAR))]
    labels=['clean','actual MuJoCo render']
    for v in VARIANTS:
        proxy=WorldTextureProxy(root/'calibration.npz','cpu',root.parent/'render_proxy_scale/fit_13_101/fit.json',v)
        with torch.no_grad():im=proxy(q,k).numpy()[0].transpose(1,2,0)
        images.append(im);labels.append('proxy MAC='+v)
    fig,axes=plt.subplots(2,5,figsize=(12,5))
    for ax,img,label in zip(axes.flat,images,labels):
        ax.imshow(img,interpolation='nearest');ax.set_title(label,fontsize=10);ax.axis('off')
    fig.suptitle('Same random world texture, fixed perspective and visibility; frame155')
    fig.tight_layout();fig.savefig(root/'fig_random_factorial_inputs.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
