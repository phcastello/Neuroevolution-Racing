import argparse,csv,re,math
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--seed',type=int,default=17);parser.add_argument('--ants',type=int,default=6);parser.add_argument('--iterations',type=int,default=2);args=parser.parse_args();root=args.directory
rows=list(csv.DictReader((root/'metrics.csv').open(encoding='utf-8')))
columns=4;fig,axs=plt.subplots(math.ceil(len(rows)/columns),columns,figsize=(18,max(4,math.ceil(len(rows)/columns)*4)),facecolor='#0a161d',squeeze=False)
for ax,row in zip(axs.flat,rows):
    saved=root/(row['id']+'.json')
    if saved.exists(): points=np.array(json.loads(saved.read_text(encoding='utf-8'))['definition']['control_points'])
    else:
        source=(root/(row['id']+'.ron')).read_text(encoding='utf-8').split('control_points: [')[1].split('    ],')[0]
        points=np.array([tuple(map(float,x)) for x in re.findall(r'\((-?[\d.eE+]+),\s*(-?[\d.eE+]+)\)',source)])
    a=np.roll(points,1,axis=0);c=np.roll(points,-1,axis=0);ab=points-a;bc=c-points
    curvature=2*np.abs(ab[:,0]*bc[:,1]-ab[:,1]*bc[:,0])/np.maximum(np.linalg.norm(ab,axis=1)*np.linalg.norm(bc,axis=1)*np.linalg.norm(c-a,axis=1),1e-8)
    points=np.vstack([points,points[0]])
    ax.set_facecolor('#0a161d');ax.plot(points[:,0],points[:,1],color='#46e1b6',linewidth=3);ax.scatter(*points[0],color='#ffd166',s=25,zorder=5)
    for i,k in enumerate(curvature):
        if k>0.019: ax.plot(points[i:i+2,0],points[i:i+2,1],color='#ff786e',linewidth=3)
    ax.set_aspect('equal');ax.axis('off');ax.set_title(row['id'].replace('generated_','')+'\nL='+row['length']+'  straight='+row['longest_straight']+'  H='+row['hairpins']+' C='+row['chicanes']+' S='+row['s_bends'],color='white',fontsize=10)
for ax in axs.flat[len(rows):]:ax.axis('off')
fig.suptitle(f'ACO sequencial | seed {args.seed} | {args.ants} formigas x {args.iterations} iterações | {len(rows)}/{len(rows)} voltas completas\nLinha central; trechos de raio 50 destacados em vermelho',color='white',fontsize=18)
fig.tight_layout(rect=(0,0,1,.95))
for ext in ['png','svg']:fig.savefig(root/('circuits.'+ext),dpi=140,facecolor=fig.get_facecolor())
print(root/'circuits.png')
