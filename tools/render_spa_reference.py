"""Plot the repository's Spa spline beside the explicit physical R50 reference."""
from pathlib import Path
import math,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path(__file__).resolve().parents[1]
s=(repo/'assets/tracks/spa.ron').read_text(encoding='utf-8').split('control_points: [')[1]
p=np.array([tuple(map(float,x)) for x in re.findall(r'\((-?[\d.]+),\s*(-?[\d.]+)\)',s)])
q=[]
for i in range(len(p)):
    a,b,c,d=[p[j%len(p)] for j in (i-1,i,i+1,i+2)]
    for t in np.arange(12)/12:
        q.append(.5*(2*b+(-a+c)*t+(2*a-5*b+4*c-d)*t*t+(-a+3*b-3*c+d)*t*t*t))
q=np.array(q);ks=[]
for i in range(36,132):
    a,b,c=q[i-1:i+2];ab=b-a;bc=c-b
    k=2*abs(ab[0]*bc[1]-ab[1]*bc[0])/(np.linalg.norm(ab)*np.linalg.norm(bc)*np.linalg.norm(c-a));ks.append((k,i))
k,index=max(ks);a,b,c=q[index-1:index+2]
center=np.linalg.solve(2*np.array([b-a,c-a]),np.array([b@b-a@a,c@c-a@a]));radius=1/k
fig,axes=plt.subplots(1,2,figsize=(13,6),facecolor='#0a161d')
for ax in axes:ax.set_facecolor('#0a161d');ax.set_aspect('equal');ax.tick_params(colors='white');[sp.set_color('#577681') for sp in ax.spines.values()]
ax=axes[0];ax.plot(q[36:133,0],q[36:133,1],color='#46e1b6',linewidth=3);ax.plot(p[3:12,0],p[3:12,1],'o',color='#ffd166')
for i in range(3,12):ax.annotate(str(i),p[i],xytext=(6,6),textcoords='offset points',color='white')
theta=np.linspace(0,2*math.pi,150);ax.plot(center[0]+radius*np.cos(theta),center[1]+radius*np.sin(theta),'--',color='#ff786e');ax.scatter(*b,color='#ff786e',s=50)
ax.set_title(f'Spa existente: controles 3 a 11\nCurvatura máxima {k:.5f} / raio {radius:.2f}',color='white');ax.set_xlabel('Unidades do simulador',color='white')
ax=axes[1];theta=np.linspace(0,math.pi,160)
for r,color,style in [(24,'#577681','-'),(50,'#46e1b6','-'),(76,'#577681','-')]:
    ax.plot(r*np.sin(theta),50-r*np.cos(theta),style,color=color,linewidth=3 if r==50 else 2)
for offset in [-26,0,26]:
    ax.plot([-150,0],[offset,offset],color='#46e1b6' if offset==0 else '#577681')
    ax.plot([-150,0],[100+offset,100+offset],color='#46e1b6' if offset==0 else '#577681')
ax.fill([42.5,57.5,57.5,42.5],[36,36,64,64],color='#ffd166',alpha=.8,label='Carro 15 × 28')
ax.annotate('',xy=(-40,0),xytext=(-90,0),arrowprops=dict(arrowstyle='->',color='white',lw=2));ax.annotate('',xy=(-90,100),xytext=(-40,100),arrowprops=dict(arrowstyle='->',color='white',lw=2))
ax.set_title('Referência explícita: hairpin de 180°, R50\nLargura 52; volta completada na Avian2D',color='white');ax.set_xlabel('Unidades do simulador',color='white');ax.legend(facecolor='#0a161d',labelcolor='white',loc='upper left')
fig.suptitle('Calibração de curvatura instantânea — La Source na aproximação 2D do projeto',color='white',fontsize=15);fig.tight_layout(rect=(0,0,1,.92))
out=repo/'docs/validation/aco_sequential/la_source_calibration.png';fig.savefig(out,dpi=150,facecolor=fig.get_facecolor());print(out)
