import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path('/Users/cril/tanmoy/research/data'); RUN=ROOT/'grounded_vision_runs/cub_bound_real_full'; CUB=ROOT/'CUB_200_2011/CUB_200_2011'
sys.path.insert(0,'/Users/cril/tanmoy/research/Natural Language Explanation/grounded_vision/src')
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.pyxai_backend import PyXAIBackend

d=np.load(RUN/'train_features.npz'); fmap=d['feature_maps']; y=d['labels']; ids=d['sample_ids']
# Explicit activation representation: 4x4 spatial tiles x 384 DINO channels.
x=fmap.reshape(len(fmap),384,4,4,4,4).mean(axis=(4,5)).transpose(0,2,3,1).reshape(len(fmap),-1)
names=[f'tile_r{r}_c{c}:dino_ch{k}' for r in range(4) for c in range(4) for k in range(384)]
tree=CertifiableTreeHead(max_depth=6,min_samples_leaf=2,feature_names=names,class_names=[str(i) for i in range(200)])
tree.fit(x,y); pred=tree.predict(x); backend=PyXAIBackend(tree)
paths={int(a):b for a,b in (line.split(maxsplit=1) for line in (CUB/'images.txt').read_text().splitlines())}
records=[]
for i,(sid,label,p) in enumerate(zip(ids,y,pred)):
    exp=backend.explain_sufficient(x[i],int(p),int(sid)); grounded=[]
    for c in exp.conditions:
        tile,chan=divmod(c.feature_idx,384); r,col=divmod(tile,4)
        grounded.append({'condition':c.to_dict(),'tile':{'row':r,'col':col,'patch_rows':[r*4,(r+1)*4-1],'patch_cols':[col*4,(col+1)*4-1],'image_box_normalized':[col/4,r/4,(col+1)/4,(r+1)/4]},'activation_channel':chan})
    contrast=backend.explain_contrastive(x[i],int(p),sample_id=int(sid))
    contrast_record=None if contrast is None else {'contrast_class':int(contrast.contrast_class),'conditions_to_alter':[c.to_dict() for c in contrast.conditions_to_alter]}
    records.append({'sample_id':int(sid),'label':int(label),'prediction':int(p),'correct':bool(label==p),'certificate':exp.to_dict(),'contrastive':contrast_record,'grounding':grounded,'image':paths[int(sid)]})
with (RUN/'activation_tree_grounded_explanations.jsonl').open('w') as f:
    for r in records: f.write(json.dumps(r, default=lambda v: int(v) if isinstance(v, np.integer) else float(v))+'\n')
chosen=[]
for cls in sorted(set(y)):
    chosen.extend([r for r in records if r['label']==cls and r['correct']][:2])
fig,axs=plt.subplots(2,4,figsize=(16,8)); axs=axs.ravel()
for ax,r in zip(axs,chosen):
    im=Image.open(CUB/'images'/r['image']).convert('RGB'); ax.imshow(im); w,h=im.size
    for g in r['grounding']:
        b=g['tile']['image_box_normalized']; ax.add_patch(plt.Rectangle((b[0]*w,b[1]*h),(b[2]-b[0])*w,(b[3]-b[1])*h,fill=False,ec='#ff7f0e',lw=2))
    ax.set_title(f"id {r['sample_id']} | y={r['label']} | {r['certificate']['sparsity']} sufficient conditions",fontsize=9); ax.axis('off')
for ax in axs[len(chosen):]: ax.axis('off')
fig.suptitle('Activation-tree sufficient reasons: orange boxes are certified DINO activation tiles',fontweight='bold'); fig.tight_layout(); fig.savefig(RUN/'activation_tree_grounded_examples.png',dpi=180)
stats={'n_images':len(y),'accuracy':float(np.mean(pred==y)),'mean_reason_size':float(np.mean([r['certificate']['sparsity'] for r in records])),'verified_rate':float(np.mean([r['certificate']['certificate']['status']=='verified' for r in records])),'pyxai_detected':backend.pyxai_available}
(RUN/'activation_tree_grounded_metrics.json').write_text(json.dumps(stats,indent=2)); print(json.dumps(stats))
