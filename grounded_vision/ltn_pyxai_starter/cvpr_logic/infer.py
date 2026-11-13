"""Infer from saved reader/head; explain without consulting concept labels."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .model import Model
from .train import outputs
from .verify import Explainer, allowed_values, predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True,help='NPZ containing x and optional ids')
    p.add_argument('--out',type=Path,default=Path('predictions.jsonl'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-length',type=int,default=4)
    p.add_argument('--max-candidates',type=int,default=5000)
    a=p.parse_args(); torch.set_num_threads(2)
    checkpoint=torch.load(a.run/'model.pt',map_location=a.device,weights_only=True)
    if checkpoint['oracle_concepts']:
        p.error('Oracle runs are diagnostics; use a learned-reader checkpoint for deployment')
    kwargs=checkpoint['model_kwargs']; kwargs['device']=a.device
    model=Model(**kwargs).to(a.device)
    if checkpoint.get('precomputed_concepts',False): model.reader=torch.nn.Identity()
    model.load_state_dict(checkpoint['state_dict']); model.head.restore_wiring(); model.eval()
    with np.load(a.data,allow_pickle=False) as z:
        x=z['x'].astype('float32'); ids=z['ids'] if 'ids' in z else np.arange(len(x)).astype(str)
    with np.load(a.run/'preprocessing.npz',allow_pickle=False) as z:
        x=(x-z['mean'])/z['std']; names=z['concept_names'].tolist()
    if not np.isfinite(x).all(): raise ValueError('Nonfinite inputs')
    if checkpoint.get('precomputed_concepts',False) and not ((x>=0)&(x<=1)).all():
        raise ValueError('Precomputed concept inputs must be probabilities in [0,1]')
    q,z=outputs(model,x,a.device)
    circuit=json.loads((a.run/'circuit.json').read_text())
    # Compare checkpoint-derived circuit too, catching changed/replaced exports.
    if model.head.export(names)!=circuit: raise RuntimeError('Checkpoint/export mismatch')
    predictions=predict(circuit,q>=.5)
    if not np.array_equal(predictions,z.argmax(1)): raise RuntimeError('Prediction mismatch')
    exp=Explainer(circuit); rows=[]
    try:
        for i,probs in enumerate(q):
            r,status=exp.shortest(probs>=.5,allowed_values(probs,checkpoint['threshold']),a.max_length,a.max_candidates)
            rows.append({'id':str(ids[i]),'prediction':int(predictions[i]),'concept_scores':probs.tolist(),
                'reason':r,'status':status,'literals':None if r is None else
                [{'concept':names[abs(l)-1],'value':bool(l>0)} for l in r]})
    finally: exp.close()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(f'Wrote {len(rows)} predictions to {a.out}')


if __name__=='__main__': main()
