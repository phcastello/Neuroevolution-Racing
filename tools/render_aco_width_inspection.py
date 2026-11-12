"""Physical-scale views of the fixed-seed width/rejection battery."""
import csv
import json
from pathlib import Path
import argparse
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("directory",type=Path)
args=parser.parse_args();root=args.directory
rows=json.loads((root/"results.json").read_text(encoding="utf-8"))
plt.rcParams.update({"figure.facecolor":"#101d26","axes.facecolor":"#101d26","text.color":"white","axes.labelcolor":"white","xtick.color":"white","ytick.color":"white"})
def read(row):
    return json.loads((root/f'{row["variant"]}-{row["seed"]}'/f'{row["id"]}.json').read_text(encoding="utf-8"))["track"]
def road(ax,t,transform=lambda p:p):
    left,right=[transform(np.array(t[k])) for k in ("left","right")]
    n=len(left)
    quads=np.array([[left[i],right[i],right[(i+1)%n],left[(i+1)%n]] for i in range(n)])
    ax.add_collection(PolyCollection(quads,facecolor="#59716d",edgecolor="none"))
    for border in (left,right):
        delta=np.roll(border,-1,axis=0)-border
        f=delta/np.maximum(np.linalg.norm(delta,axis=1)[:,None],1e-8)
        normal=np.c_[-f[:,1],f[:,0]]*7
        walls=np.stack([border+normal,border-normal,np.roll(border,-1,axis=0)-normal,np.roll(border,-1,axis=0)+normal],axis=1)
        ax.add_collection(PolyCollection(walls,facecolor="#d5e8dd",edgecolor="none"))
    ax.set_aspect("equal")
    ax.grid(alpha=.12)
interlagos=json.loads((root/"interlagos.json").read_text(encoding="utf-8"))
wide=[r for r in rows if r["variant"]=="wide" and r.get("id")]
previous=[r for r in rows if r["variant"]=="previous" and r.get("id")]
examples=[("Interlagos",interlagos)]+[(f's{r["seed"]} i{r["iteration"]} a{r["ant"]} · largura {r["width"]:.0f}',read(r)) for r in wide[:5]]
span=max(np.ptp(np.array(t["centerline"]),axis=0).max() for _,t in examples)*.58
fig,axes=plt.subplots(2,3,figsize=(16,11))
for ax,(name,t) in zip(axes.flat,examples):
    p=np.array(t["centerline"]);center=(p.max(axis=0)+p.min(axis=0))/2
    road(ax,t,lambda p:p-center)
    ax.set(xlim=(-span,span),ylim=(-span,span),xlabel="unidades do simulador",ylabel="unidades do simulador")
    ax.set_title(f'{name}\n{t["definition"]["width"]/15:.2f} carros nominais · {(t["definition"]["width"]-14)/15:.2f} entre muros',fontsize=11)
fig.suptitle("Interlagos e ACO largo: mesma escala física em todos os painéis",fontsize=17)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(root/"same-scale.png",dpi=135);plt.close(fig)

samples=[("Interlagos",interlagos),("ACO anterior",read(previous[0])),("ACO largo",read(next(r for r in wide if r["width"]==100)))]
fig,axes=plt.subplots(1,3,figsize=(14,5))
for ax,(name,t) in zip(axes,samples):
    p=np.array(t["centerline"]);a=p-np.roll(p,1,axis=0);b=np.roll(p,-1,axis=0)-p
    k=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-8)
    i=int(np.argmin(k));f=(a[i]+b[i]);f/=np.linalg.norm(f);normal=np.array([-f[1],f[0]])
    def transform(points):
        q=points-p[i];return np.c_[q@f,q@normal]
    road(ax,t,transform)
    width=t["definition"]["width"]
    for j in range(-2,3):
        y=j*16
        car=np.array([[-14,y-7.5],[14,y-7.5],[14,y+7.5],[-14,y+7.5]])
        fits=abs(y)+7.5<(width-14)/2
        ax.add_patch(Polygon(car,facecolor="#3bdeac" if fits else "#fa776b",edgecolor="#111111",linewidth=1))
    ax.set(xlim=(-75,75),ylim=(-75,75),xlabel="unidades do simulador",ylabel="unidades do simulador")
    ax.set_title(f'{name}\nnominal {width:.0f} · entre faces dos muros ≈ {width-14:.0f}\ncarros reais: 28 × 15; verdes cabem',fontsize=11)
fig.suptitle("Seção ampliada na mesma escala: asfalto, muros físicos de 14 e cinco carros",fontsize=15)
fig.tight_layout(rect=(0,0,1,.90));fig.savefig(root/"width-closeup.png",dpi=140);plt.close(fig)

hairpin=json.loads((root/"hairpin.json").read_text(encoding="utf-8"))
fig,ax=plt.subplots(figsize=(12,5));road(ax,hairpin);p=np.array(hairpin["centerline"])
ax.set(xlim=(p[:,0].min()-100,p[:,0].max()+100),ylim=(p[:,1].min()-100,p[:,1].max()+100),xlabel="unidades do simulador",ylabel="unidades do simulador")
ax.set_title("Hairpin de 180°: raio central 75 · largura 100 · faixa nominal interna de raio 25")
fig.tight_layout();fig.savefig(root/"wide-hairpin.png",dpi=140);plt.close(fig)
summary={}
for variant in ("previous","wide"):
    subset=[r for r in rows if r["variant"]==variant]
    valid=[r for r in subset if r.get("id")]
    reasons={}
    for r in subset:
        key=r.get("reason") or "accepted";reasons[key]=reasons.get(key,0)+1
    summary[variant]={"attempted":len(subset),"valid":len(valid),"reference_completed":sum(r["reference_completed"] for r in valid),"accepted":sum(r["accepted"] for r in valid),"reasons":reasons}
    for metric in ("width","nominal_cars","useful_width_mean","useful_width_min","physical_cars"):
        summary[variant][metric]=float(np.mean([r[metric] for r in valid]))
    for metric in ("rectangularity","heading_entropy","complexity"):
        summary[variant][metric]=float(np.mean([r["metrics"]["organic"][metric] for r in valid]))
(root/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
columns=["variant","seed","iteration","ant","id","width","nominal_cars","useful_width_mean","useful_width_min","physical_cars","accepted","reason","reference_completed","reference_ticks"]
with (root/"metrics.csv").open("w",newline="",encoding="utf-8") as out:
    w=csv.DictWriter(out,fieldnames=columns);w.writeheader()
    for r in rows:w.writerow({k:r.get(k) for k in columns})
if (root/"corridor-rejected/diagnostic.json").exists():
    d=json.loads((root/"corridor-rejected/diagnostic.json").read_text(encoding="utf-8"))["snapshot"]["diagnostic"]
    fig,ax=plt.subplots(figsize=(10,6));p=np.array(d["partial"]);ax.plot(*p.T,color="#53dec0",lw=3)
    for a in d["terminal"]["alternatives"]:
        q=np.array(a["points"])
        if len(q)>1:ax.plot(*q.T,color="#fa776b" if a["rejection"] else "#88999a",lw=.8,alpha=.5)
    for issue in d["issues"]:
        q=np.array(issue["points"])
        if len(q):ax.scatter(*q.T,color="red",s=35)
    ax.set_aspect("equal");ax.set_title("Diagnóstico de alternativa realmente rejeitada por distância entre corredores\nverde: trajetória parcial · vermelho: conflitos · não é um circuito treinável")
    fig.tight_layout();fig.savefig(root/"corridor-diagnostic.png",dpi=140);plt.close(fig)
print(json.dumps(summary,indent=2))
