"""Nearest-neighbour exemplar grounding for certified DINO tiles."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

root=Path('/Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011'); run=Path('/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full')
d=np.load(run/'train_features.npz'); fm=d['feature_maps']; ids=d['sample_ids']; labels=d['labels']
emb=fm.reshape(len(fm),384,4,4,4,4).mean(axis=(4,5)).transpose(0,2,3,1) # n,r,c,d
emb/=np.maximum(np.linalg.norm(emb,axis=-1,keepdims=True),1e-8)
index={int(s):i for i,s in enumerate(ids)}; paths={int(x.split()[0]):x.split(maxsplit=1)[1] for x in (root/'images.txt').read_text().splitlines()}
def crop(sid,r,c):
 im=Image.open(root/'images'/paths[sid]).convert('RGB'); w,h=im.size
 return im.crop((c*w//4,r*h//4,(c+1)*w//4,(r+1)*h//4))
src=run/'activation_part_grounded_nlx.jsonl'; dst=run/'activation_exemplar_grounded_nlx.jsonl'; examples=[]
with src.open() as fi,dst.open('w') as fo:
 for line in fi:
  rec=json.loads(line); i=index[rec['sample_id']]
  for g in rec['grounding']:
   r,c=g['tile']['row'],g['tile']['col']; sim=emb[:,r,c]@emb[i,r,c]; sim[i]=-np.inf; j=int(np.argmax(sim))
   g['exemplar']={'sample_id':int(ids[j]),'label':int(labels[j]),'image':paths[int(ids[j])],'tile':{'row':r,'col':c},'cosine_similarity':float(sim[j])}
  rec['nlx_exemplar']={'text':rec['nlx_part_only']['text']+' Matching reference crops are attached for each certified tile.','claim':'Exemplars are nearest non-self DINO tile embeddings; no semantic attribute is inferred.'}
  fo.write(json.dumps(rec)+'\n'); examples.append(rec)
fig,axs=plt.subplots(4,4,figsize=(12,12)); axs=axs.ravel()
for k,rec in enumerate(examples[:8]):
 g=rec['grounding'][0]; r,c=g['tile']['row'],g['tile']['col']; ex=g['exemplar']
 axs[2*k].imshow(crop(rec['sample_id'],r,c)); axs[2*k].set_title(f"query {rec['sample_id']}\n{g['validated_cub_parts'] or ['unannotated']}",fontsize=8); axs[2*k].axis('off')
 axs[2*k+1].imshow(crop(ex['sample_id'],r,c)); axs[2*k+1].set_title(f"exemplar {ex['sample_id']}\ncos={ex['cosine_similarity']:.2f}",fontsize=8); axs[2*k+1].axis('off')
fig.suptitle('Certified-tile exemplar grounding: query (left) and nearest non-self DINO tile (right)',fontweight='bold'); fig.tight_layout(); fig.savefig(run/'activation_exemplar_examples.png',dpi=180)
print(dst)
