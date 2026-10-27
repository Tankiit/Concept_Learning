import json,sys
from pathlib import Path
import numpy as np
import shap
sys.path.insert(0,'/Users/cril/tanmoy/research/Natural Language Explanation/grounded_vision/src')
from grounded_vision.reasoning.tree_head import CertifiableTreeHead
run=Path('/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full');d=np.load(run/'train_features.npz');fm=d['feature_maps'];y=d['labels']
x=fm.reshape(len(fm),384,4,4,4,4).mean((4,5)).transpose(0,2,3,1).reshape(len(fm),-1)
names=[f'tile_r{r}_c{c}:dino_ch{k}' for r in range(4) for c in range(4) for k in range(384)];t=CertifiableTreeHead(max_depth=6,min_samples_leaf=2,feature_names=names);t.fit(x,y);p=t.predict(x)
sv=shap.TreeExplainer(t.clf).shap_values(x)
if isinstance(sv,list): vals=np.stack(sv,axis=1)[np.arange(len(x)),p]
else: vals=sv[np.arange(len(x)),:,p] if sv.ndim==3 else sv
selected=json.load(open(run/'grounding_selected_reasons.json'));sel=[{c['feature_idx'] for c in r['conditions']} for r in selected]
rows=[]
for i in range(len(x)):
 k=len(sel[i]); top=set(np.argsort(np.abs(vals[i]))[-k:]); rows.append({'sample_id':int(d['sample_ids'][i]),'k':k,'overlap_with_grounded_reason':len(top&sel[i])/k,'shap_feature_indices':sorted(map(int,top))})
result={'method':'TreeSHAP on the same activation decision tree','n_images':len(x),'mean_reason_size':float(np.mean([r['k'] for r in rows])),'mean_shap_grounded_overlap':float(np.mean([r['overlap_with_grounded_reason'] for r in rows])),'interpretation':'SHAP ranks attribution; it does not guarantee sufficiency. Grounded reasons are verifier-sufficient subsets.' ,'per_image':rows}
json.dump(result,open(run/'shap_vs_sufficient_results.json','w'),indent=2);print(result['mean_shap_grounded_overlap'])
