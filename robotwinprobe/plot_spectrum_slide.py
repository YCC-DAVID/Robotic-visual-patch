"""PPT figure from existing measured spectrum summary; CPU only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'robotwinprobe/out/spectrum_20261028/summary.json'
data=json.loads(source.read_text())
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':15,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A3ADB8','axes.labelcolor':'#263649','xtick.color':'#526174','ytick.color':'#526174','axes.unicode_minus':False,'svg.fonttype':'path'})
ks=np.array([1,2,3,4,6,8,11,16,23,32,45,60])
fig=plt.figure(figsize=(16,9),facecolor='white')
fig.text(.065,.925,'成像链衰减高频，增大贴纸可保留更多纹理周期',fontsize=25,weight='bold',color='#172B45')
fig.text(.065,.873,'RoboTwin 实测｜正弦条纹输入 → SAPIEN 渲染 → 缩放至模型输入（224）',fontsize=16,color='#526174')
a=fig.add_axes([.075,.29,.395,.47]);b=fig.add_axes([.565,.29,.37,.47])
colors=['#2280B9','#E17B26','#238B77']
for ax in (a,b):
 ax.set_xscale('log',base=2);ax.set_xlim(1,60);ax.set_ylim(0,1.05)
 ax.set_xticks([1,2,4,8,16,32,60]);ax.set_xticklabels(['1','2','4','8','16','32','60'])
 ax.set_yticks([0,.25,.5,.75,1]);ax.set_yticklabels(['0%','25%','50%','75%','100%'])
 ax.grid(axis='y',color='#E6EBF0');ax.set_axisbelow(True)
 ax.axhline(.5,color='#8A97A7',linestyle=(0,(3,3)),linewidth=1)
 ax.set_xlabel('纹理频率（周期 / patch 边长）',labelpad=12)
 ax.set_ylabel('条纹振幅保留比例',labelpad=8)
for size,color in zip(['0.06','0.10','0.14'],colors):
 s=data['sizes'][size];cut=s['cutoff_k']['model_real_mean_0.5']
 a.plot(ks,s['mtf']['model_real_mean'],color=color,lw=2.7,marker='o',ms=4,label=f'{int(float(size)*100)} cm  ·  50% 点 ≈ {cut:.1f} 周期')
 a.scatter([cut],[.5],color=color,s=55,zorder=5,edgecolor='white',linewidth=1)
a.set_title('A   不同尺寸：模型实际接收的信号',loc='left',fontsize=17,pad=18,weight='bold')
a.legend(loc='upper right',frameon=False,fontsize=12)
s=data['sizes']['0.10']
b.plot(ks,s['mtf']['native_real_mean'],color='#667A94',lw=2.7,marker='o',ms=4,label='相机原生图像（320 × 240）')
b.plot(ks,s['mtf']['model_real_mean'],color=colors[1],lw=2.7,marker='o',ms=4,label='模型输入（缩放至 224）')
b.set_title('B   10 cm：输入缩放进一步削弱细节',loc='left',fontsize=17,pad=18,weight='bold')
b.legend(loc='upper right',frameon=False,fontsize=12)
fig.text(.075,.155,'50% 振幅点约为 0.9 周期 / cm（周期长度约 1.1 cm）',fontsize=19,weight='bold',color='#172B45')
fig.text(.075,.108,'这是平滑衰减，不是硬截止；高频残留或混叠不代表原始细节被忠实传递。',fontsize=14,color='#526174')
fig.text(.075,.045,'条件：place_container_plate · site07 · 6 个初始状态 · 头部相机 · 标准光照 · yaw = 0\n曲线：两轴、两相位及各状态测量增益的中位数。仅描述成像传递，不代表闭环攻击效果或通用尺寸阈值。',fontsize=11,color='#697889',linespacing=1.5)
out=ROOT/'result/figures/robotwin_low_frequency_ppt';out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out.with_suffix('.png'),dpi=240,facecolor='white')
fig.savefig(out.with_suffix('.svg'),facecolor='white')
print(out.with_suffix('.png'))
