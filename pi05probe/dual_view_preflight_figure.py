"""Recreate the E26 four-frame preflight comparison from recorded arrays."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    r=Path(__file__).resolve().parent/'out/dual_view'
    d=np.load(r/'preflight/inputs.npz');p=np.load(r/'preflight/proxy_images.npz')
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig,axes=plt.subplots(4,3,figsize=(10,13),constrained_layout=True)
    for row,k in enumerate([0,8,16,24]):
        for col,im in enumerate([d['check__base'][k],d['check__wrist'][k],p['check__wrist'][k]]):
            axes[row,col].imshow(im,interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['真实base','真实wrist','wrist成像代理'][col],fontproperties=font)
        axes[row,0].set_ylabel('init{} step{}'.format(d['ep_id'][k],d['env_step'][k]))
    fig.suptitle('未用于拟合的平滑随机纹理：逐状态双相机核验',fontproperties=font,fontsize=14)
    fig.savefig(r/'preflight_views.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
