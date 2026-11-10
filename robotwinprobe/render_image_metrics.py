"""Frequency and paired input diagnostics, with no model weights loaded."""
import json
import numpy as np
import bootstrap as B
B.model_paths()
from render_proxy import Pool
from render_geometry import CAMERAS,WIRE


def main():
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from openpi_client.image_tools import resize_with_pad
    torch.set_num_threads(4);root=B.OUT/'render_transfer'
    pool=Pool([root/'calibration'/f'seed{s}.npz' for s in [10110,10111]],'cpu',global_response=True)
    rows=[];frequency={};examples={}
    for variant in ['full','band32']:
        for method in ['digital','proxy']:
            label=method+'_'+variant;rgb=np.array(Image.open(root/f'train_{label}/texture_0500.png').convert('RGB'))
            q=torch.tensor(rgb).permute(2,0,1)[None].float()/255
            x=rgb.astype(float)/255;x-=x.mean((0,1),keepdims=True)
            power=abs(np.fft.fft2(x,axes=(0,1)))**2;freq=np.fft.fftfreq(128)
            high=np.maximum(abs(freq[:,None]),abs(freq[None,:]))>.125
            frequency[label]={'energy_outside_central_32_frequency_square':float(power[high].sum()/max(power.sum(),1e-20)),
                              'std_RGB_after_removing_channel_means':float(np.sqrt((x*x).mean()))}
            for i,frame in enumerate(pool.frames):
                with np.load(root/f'capture_{variant}'/f"seed{frame['seed']}.npz") as d:
                    j=int(np.flatnonzero(d['query']==frame['query'])[0])
                    for name in CAMERAS:
                        v=frame['views'][name]
                        mask=F.pad(F.interpolate(v['mask'],(168,224),mode='nearest'),(0,0,28,28))[0,0].numpy().astype(bool)
                        interior=mask[1:-1,1:-1]&mask[:-2,1:-1]&mask[2:,1:-1]&mask[1:-1,:-2]&mask[1:-1,2:]
                        images={'digital':pool.image(q,i,name,'digital')[0].permute(1,2,0).numpy(),
                                'proxy':pool.image(q,i,name,'proxy')[0].permute(1,2,0).numpy(),
                                'render':resize_with_pad(d[name+'_'+label][j],224,224).astype(float)/255}
                        row={'texture':label,'seed':frame['seed'],'query':frame['query'],'camera':name,'interior_pixels':int(interior.sum()),'metrics':{}}
                        if interior.sum()>=16:
                            for mode,image in images.items():
                                samples=image[1:-1,1:-1][interior]
                                contrast=float(np.sqrt(((samples-samples.mean(0))**2).mean()))
                                gx=(image[1:-1,2:]-image[1:-1,:-2])/2;gy=(image[2:,1:-1]-image[:-2,1:-1])/2
                                sharpness=float(np.sqrt((gx[interior]**2+gy[interior]**2).mean()))
                                row['metrics'][mode]={'RGB_contrast':contrast,'spatial_gradient_RMS':sharpness,
                                                      'gradient_over_contrast':sharpness/max(contrast,1e-10)}
                            rows.append(row)
                        if i==0 and name=='head_camera':
                            examples[label+'_texture']=rgb
                            examples[label+'_inputs']=np.array([v['clean224'][0].permute(1,2,0).numpy(),images['digital'],images['render'],images['proxy']])
                            examples[label+'_mask']=mask
    (root/'image_metrics.json').write_text(json.dumps({'frequency':frequency,'rows':rows,
        'note':'Interior gradient/contrast describes visible sharpness, not attack efficacy. Band32 is bilinear parameterization, not a strict spectral cutoff.'},indent=2)+'\n')
    np.savez_compressed(root/'learned_figure_data.npz',**examples)


if __name__=='__main__':main()
