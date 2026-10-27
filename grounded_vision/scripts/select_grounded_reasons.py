import itertools,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/Users/cril/tanmoy/research/Natural Language Explanation/grounded_vision/src')
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
from grounded_vision.explanations.tree_verifier import TreeSufficiencyVerifier
from grounded_vision.explanations.schema import ExplanationCondition,VerificationStatus
root=Path('/Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011'); run=Path('/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full')
d=np.load(run/'train_features.npz'); fm=d['feature_maps']; y=d['labels']; ids=d['sample_ids']; x=fm.reshape(len(fm),384,4,4,4,4).mean((4,5)).transpose(0,2,3,1).reshape(len(fm),-1)
names=[f'tile_r{r}_c{c}:dino_ch{k}' for r in range(4) for c in range(4) for k in range(384)]; t=CertifiableTreeHead(max_depth=6,min_samples_leaf=2,feature_names=names);t.fit(x,y); v=TreeSufficiencyVerifier(t.clf,names)
parts={}; pnames={int(z.split()[0]):z.split(maxsplit=1)[1] for z in (root/'parts/parts.txt').read_text().splitlines()}
for z in (root/'parts/part_locs.txt').read_text().splitlines():
 s,p,a,b,vis=z.split();
 if int(vis): parts.setdefault(int(s),[]).append((pnames[int(p)],float(a),float(b)))
imgs={int(z.split()[0]):z.split(maxsplit=1)[1] for z in (root/'images.txt').read_text().splitlines()}
from PIL import Image
out=[]
for xi,sid in zip(x,ids):
 pred=int(t.predict(xi[None])[0]); path=[]
 for q in t.get_decision_path(xi): path.append(ExplanationCondition(q.feature_idx,q.feature_name,'<=' if q.is_left else '>',q.threshold,q.value))
 valid=[]
 for n in range(1,len(path)+1):
  for sub in itertools.combinations(path,n):
   if v.verify_sufficiency(list(sub),pred).status==VerificationStatus.VERIFIED: valid.append(list(sub))
 bestn=min(map(len,valid)); cand=[a for a in valid if len(a)<=bestn+1]; im=Image.open(root/'images'/imgs[int(sid)]);w,h=im.size
 def score(a):
  found=set()
  for c in a:
   tile=c.feature_idx//384;r,col=divmod(tile,4)
   found|={n for n,px,py in parts.get(int(sid),[]) if col/4<=px/w<=(col+1)/4 and r/4<=py/h<=(r+1)/4}
  return len(found)
 chosen=max(cand,key=score);out.append({'sample_id':int(sid),'prediction':pred,'shortest_size':bestn,'selected_size':len(chosen),'part_score':score(chosen),'conditions':[c.to_dict() for c in chosen]})
json.dump(out,open(run/'grounding_selected_reasons.json','w'),indent=2);print({'mean_shortest':np.mean([a['shortest_size'] for a in out]),'mean_selected':np.mean([a['selected_size'] for a in out]),'mean_part_score':np.mean([a['part_score'] for a in out])})
