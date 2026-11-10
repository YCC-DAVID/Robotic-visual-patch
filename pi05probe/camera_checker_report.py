"""Separate resolvable perspective geometry from actual224 sampling artifacts."""
import json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import cv2
    dest=Path(__file__).resolve().parent/'out/camera_checker'
    a=np.load(dest/'inputs.npz',allow_pickle=False)
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    mask=a['visible1024'][0]
    y,x=np.where(mask);y0,y1=max(0,y.min()-16),min(1024,y.max()+17);x0,x1=max(0,x.min()-16),min(1024,x.max()+17)
    sy,sx=slice(y0,y1),slice(x0,x1)
    titles=['① 世界纹理 Q128','② 相机参数正向透视\n1024，无拟合映射','③ 真实 MuJoCo\n1024，无光照 nearest','④ 真实 MuJoCo\n1024，正常光照']
    fig,axes=plt.subplots(4,4,figsize=(16,12),constrained_layout=True)
    for row,(name,s) in enumerate(zip(a['patterns'],a['sizes'])):
        bg=a[name+'__lit1024'][0]
        digital=np.where(mask[...,None],a[name+'__projected1024'][0],bg)
        minimal=np.where(mask[...,None],a[name+'__minimal1024'][0],bg)
        images=[a[name+'__Q'],digital[sy,sx],minimal[sy,sx],bg[sy,sx]]
        for col,image in enumerate(images):
            axes[row,col].imshow(image,interpolation='nearest')
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(titles[col],fontproperties=font,fontsize=12)
        axes[row,0].set_ylabel('格宽 {} texels\n{:.3f} mm'.format(int(s),float(a['patch_m'])*1000*int(s)/128),fontproperties=font,fontsize=11)
    fig.suptitle('先检查几何：四行均为同一矩形世界纹理的透视投影',fontproperties=font,fontsize=16)
    fig.savefig(dest/'geometry1024.png',dpi=160);plt.close(fig)
    # Every column below is at model224, displayed at identical nearest zoom.
    H=a['Q_to_image224_H']
    by,bx=np.where(a['mask224_legacy'][0]);s224y=slice(max(0,by.min()-4),min(224,by.max()+5));s224x=slice(max(0,bx.min()-4),min(224,bx.max()+5))
    summary=json.loads((dest/'summary.json').read_text())
    widths=summary['geometry']['cell_widths_model224']
    fig,axes=plt.subplots(4,3,figsize=(12,11),constrained_layout=True)
    for row,(name,s,w) in enumerate(zip(a['patterns'],a['sizes'],widths)):
        raw=a[name+'__model224'][0];m=a['mask224_legacy'][0].astype(bool)
        nn=cv2.warpPerspective(a[name+'__Q'],H,(224,224),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
        bilinear=cv2.warpPerspective(a[name+'__Q'],H,(224,224),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        for col,image in enumerate([np.where(m[...,None],nn,raw),np.where(m[...,None],bilinear,raw),raw]):
            axes[row,col].imshow(image[s224y,s224x],interpolation='nearest');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(['透视后 nearest224','透视后 bilinear224','模型实际输入\n真实默认256→224'][col],fontproperties=font,fontsize=12)
        axes[row,0].set_ylabel('{} texels\n远端 {:.2f}px / 近端 {:.2f}px'.format(int(s),w['far_width_px'],w['near_width_px']),fontproperties=font,fontsize=11)
    fig.suptitle('再检查采样：前两行在224输入中仅约1–3像素／格',fontproperties=font,fontsize=15)
    fig.savefig(dest/'sampling224.png',dpi=160);plt.close(fig)
    # Three time points, all geometrically projected, no screen-cut examples.
    fig,axes=plt.subplots(3,2,figsize=(9,10),constrained_layout=True)
    for j,step in enumerate(a['steps']):
        bg=a['cell16__lit1024'][j];m=a['visible1024'][j]
        ref=np.where(m[...,None],a['cell16__projected1024'][j],bg)
        for col,im in enumerate([ref,bg]):
            axes[j,col].imshow(im[sy,sx],interpolation='nearest');axes[j,col].set_xticks([]);axes[j,col].set_yticks([])
            if j==0:axes[j,col].set_title(['相机参数透视参考','真实正常render'][col],fontproperties=font,fontsize=13)
        axes[j,0].set_ylabel('环境 step '+str(step),fontproperties=font,fontsize=12)
    fig.suptitle('同一世界Q：三个时刻的投影与遮挡',fontproperties=font,fontsize=16)
    fig.savefig(dest/'temporal1024.png',dpi=160);plt.close(fig)
    print('wrote geometry1024.png, sampling224.png, temporal1024.png')


if __name__=='__main__':main()
