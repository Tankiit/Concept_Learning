"""Train a visual concept reader + difflogic head; calibrate and verify reasons."""
import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import demo, load
from .model import Model, UPSTREAM_COMMIT
from .verify import Explainer, predict, calibrate, allowed_values, semantic_truth, crosscheck_pyxai


def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def loss_fn(q, logits, c, mask, y, objective):
    if objective == 'bce':
        concept_loss = nn.functional.binary_cross_entropy(q[mask],c[mask]) if mask.any() else q.sum()*0
        return concept_loss + nn.functional.cross_entropy(logits,y)
    import ltn
    truth = torch.cat([torch.where(c[mask]==1,q[mask],1-q[mask]),
                       logits.softmax(-1).gather(1,y[:,None]).flatten()]).clamp(0,1)
    v=ltn.Variable('supervised_statement',truth)
    pred=ltn.Predicate(func=lambda t:t)
    forall=ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2),quantifier='f')
    return 1-forall(v,pred(v)).value


@torch.no_grad()
def outputs(model,x,device,hard=True):
    model.eval() if hard else model.train()
    qs,ps=[],[]
    for batch in torch.as_tensor(x).split(256):
        q,p=model(batch.to(device),hard=hard)
        qs.append(q.cpu().numpy()); ps.append(p.cpu().numpy())
    model.eval()
    return np.concatenate(qs),np.concatenate(ps)


def evaluate(circuit,q,c,y,ids,threshold,args):
    exp=Explainer(circuit,args.solver_conflicts)
    rows=[]
    # A frozen head repeats queries on the same bit patterns; cache both policies.
    cache={}
    try:
        for i in range(len(q)):
            bits=(q[i]>=.5).astype(int); box=allowed_values(q[i],threshold)
            key=(tuple(bits),)
            if key not in cache:
                cache[key]=exp.shortest(bits,max_length=args.max_length,max_candidates=args.max_candidates)
            ordinary,os=cache[key]
            key=(tuple(bits),tuple(box.flatten()))
            if key not in cache:
                cache[key]=exp.shortest(bits,box,args.max_length,args.max_candidates)
            robust,rs=cache[key]
            covered=bool(box[np.arange(len(bits)),c[i].astype(int)].all())
            rt=semantic_truth(robust,c[i]) if robust is not None else None
            if robust is not None and not rt and covered:
                raise RuntimeError('False issued premise despite concept-set coverage')
            rows.append({'id':str(ids[i]),'target':int(y[i]),'prediction':int(predict(circuit,bits)[0]),
                'scores':q[i].tolist(),'true_concepts':c[i].astype(int).tolist(),'bits':bits.tolist(),
                'ordinary_reason':ordinary,'ordinary_status':os,
                'ordinary_true':semantic_truth(ordinary,c[i]) if ordinary is not None else None,
                'robust_reason':robust,'robust_status':rs,'robust_true':rt,
                'set_covers_truth':covered,'accepted':robust is not None,
                'matched':robust is not None and ordinary is not None and len(robust)==len(ordinary)})
    finally:
        exp.close()
    accepted=[r for r in rows if r['accepted']]
    matched=[r for r in rows if r['matched']]
    mean=lambda v:float(np.mean(v)) if len(v) else None
    return rows,{'test_count':len(rows),'accepted_count':len(accepted),'matched_count':len(matched),
        'task_accuracy':mean([r['prediction']==r['target'] for r in rows]),
        'concept_accuracy':float(((q>=.5)==c).mean()),
        'joint_concept_set_coverage':mean([r['set_covers_truth'] for r in rows]),
        'explanation_coverage':len(accepted)/len(rows),
        'accepted_reason_error':mean([not r['robust_true'] for r in accepted]),
        'false_issued_reason_rate_all_test':sum(not r['robust_true'] for r in accepted)/len(rows),
        'matched_repairs':sum(not r['ordinary_true'] and r['robust_true'] for r in matched),
        'matched_spoils':sum(r['ordinary_true'] and not r['robust_true'] for r in matched),
        'paired_reason_delta':mean([int(r['robust_true'])-int(r['ordinary_true']) for r in matched]),
        'status_counts':{s:sum(r['robust_status']==s for r in rows) for s in sorted({r['robust_status'] for r in rows})}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help='NPZ cached features/images; omitted = synthetic image demo')
    p.add_argument('--out',type=Path,default=Path('results/difflogic_demo'))
    p.add_argument('--backend',choices=['reference','upstream'],default='reference')
    p.add_argument('--device',default='cpu')
    p.add_argument('--objective',choices=['bce','ltn'],default='bce')
    p.add_argument('--ste',action='store_true',help='Hard concepts forward, identity surrogate backward during training')
    p.add_argument('--oracle-concepts',action='store_true',help='Diagnostic: use complete annotated concepts, bypass reader')
    p.add_argument('--precomputed-concepts',action='store_true',help='x is frozen reader probabilities [N,M]; bypass reader')
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--depth',type=int,default=2)
    p.add_argument('--tau',type=float,default=4.)
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--alpha',type=float,default=.1)
    p.add_argument('--max-length',type=int,default=4)
    p.add_argument('--max-candidates',type=int,default=5000)
    p.add_argument('--solver-conflicts',type=int,default=100000)
    p.add_argument('--pyxai-check',action='store_true',help='Exhaustive tree cross-check, at most 10 concepts')
    args=p.parse_args()
    if args.epochs<1 or not 0<args.alpha<1 or args.max_length<0 or min(args.max_candidates,args.solver_conflicts)<1:
        p.error('Invalid epochs, alpha or search budgets')
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    d=load(args.data) if args.data else demo(args.seed)
    if args.oracle_concepts and args.precomputed_concepts:
        p.error('Choose oracle concepts or precomputed predictions, not both')
    if args.oracle_concepts:
        if not d['observed'].all():
            p.error('Oracle-concept mode requires complete concepts in all splits')
        d['x']=d['c'].copy()
    names=d['concept_names'].tolist(); m=len(names); classes=int(d['y'].max())+1
    if args.precomputed_concepts and (d['x'].shape != d['c'].shape or not ((d['x']>=0)&(d['x']<=1)).all()):
        p.error('Precomputed concepts require x in [0,1] with shape [N,M]')
    if args.pyxai_check and m>10: p.error('--pyxai-check supports at most 10 concepts')
    # Standardize cached features using training rows only. Do not scale oracle bits.
    mean=np.zeros(d['x'].shape[1:],dtype='float32'); std=np.ones_like(mean)
    if d['x'].ndim==2 and not (args.oracle_concepts or args.precomputed_concepts):
        mean=d['x'][d['split']=='train'].mean(0)
        std=d['x'][d['split']=='train'].std(0).clip(.00001)
        d['x']=(d['x']-mean)/std
    kwargs={'input_shape':list(d['x'].shape[1:]),'concepts':m,'classes':classes,
        'width':args.width,'depth':args.depth,'tau':args.tau,'backend':args.backend,'device':args.device}
    model=Model(**kwargs).to(args.device)
    if args.oracle_concepts or args.precomputed_concepts: model.reader=nn.Identity()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    splits={s:np.flatnonzero(d['split']==s) for s in ['train','tune','calibration','test']}
    best=-np.inf; checkpoint=None; history=[]; gradients=None
    for epoch in range(args.epochs):
        model.train(); losses=[]
        order=torch.randperm(len(splits['train'])).numpy()
        for j in range(0,len(order),128):
            ix=splits['train'][order[j:j+128]]
            x=torch.as_tensor(d['x'][ix],device=args.device)
            c=torch.as_tensor(d['c'][ix],device=args.device)
            mask=torch.as_tensor(d['observed'][ix],device=args.device)
            y=torch.as_tensor(d['y'][ix],device=args.device,dtype=torch.long)
            q,logits=model(x,ste=args.ste)
            loss=loss_fn(q,logits,c,mask,y,args.objective)
            opt.zero_grad(); loss.backward()
            if gradients is None:
                rp=list(model.reader.parameters())
                gradients={'reader':float(rp[0].grad.norm()) if rp else None,
                           'gates':float(model.head.layers[0].weights.grad.norm())}
            opt.step(); losses.append(float(loss.detach()))
        ix=splits['tune']; q,z=outputs(model,d['x'][ix],args.device)
        ca=float(((q>=.5)==d['c'][ix])[d['observed'][ix]].mean())
        ya=float((z.argmax(1)==d['y'][ix]).mean())
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'tune_concept_accuracy':ca,'tune_hard_task_accuracy':ya})
        if ca+ya>best:
            best=ca+ya; checkpoint=copy.deepcopy(model.state_dict())
        if epoch%20==0: print(f'epoch {epoch}: concept={ca:.3f} hard_task={ya:.3f}',flush=True)
    model.load_state_dict(checkpoint); model.head.restore_wiring(); model.eval()
    circuit=model.head.export(names)
    ix=splits['calibration']; qc,_=outputs(model,d['x'][ix],args.device)
    threshold=calibrate(qc,d['c'][ix],args.alpha)
    ix=splits['test']; qt,zt=outputs(model,d['x'][ix],args.device)
    _,soft=outputs(model,d['x'][ix],args.device,hard=False)
    hard=predict(circuit,qt>=.5)
    if not np.array_equal(hard,zt.argmax(1)): raise RuntimeError('Hard model/export mismatch')
    # Exhaustive equivalence of head outputs for small concept domains.
    domain=np.array(list(itertools.product([0,1],repeat=m))) if m<=10 else np.random.default_rng(args.seed).integers(0,2,(1024,m))
    model.eval()
    with torch.no_grad(): ph=model.head(torch.as_tensor(domain,dtype=torch.float32,device=args.device)).argmax(1).cpu().numpy()
    if not np.array_equal(ph,predict(circuit,domain)): raise RuntimeError('Discrete head/export mismatch')
    rows,metrics=evaluate(circuit,qt,d['c'][ix],d['y'][ix],d['ids'][ix],threshold,args)
    metrics.update({'initial_gradient_norms':gradients,'soft_hard_disagreement':float((soft.argmax(1)!=hard).mean()),
        'alpha':args.alpha,'calibration_threshold':threshold,'export_check_inputs':len(domain),
        'export_check_exhaustive':m<=10,'oracle_concepts':args.oracle_concepts,
        'dataset':'synthetic' if args.data is None else str(args.data),
        'scope':'exact Boolean head reasons; marginal false-issued-premise bound under joint coverage, not conditional precision or task correctness'})
    if args.pyxai_check: metrics['pyxai']=crosscheck_pyxai(circuit)
    args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'metrics.json',metrics); write_json(args.out/'training.json',history)
    write_json(args.out/'circuit.json',circuit)
    (args.out/'per_example.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    versions={n:importlib.metadata.version(n) for n in ['torch','numpy','python-sat']}
    if args.objective=='ltn': versions['LTNtorch']=importlib.metadata.version('LTNtorch')
    if args.pyxai_check: versions['pyxai']=importlib.metadata.version('pyxai')
    write_json(args.out/'provenance.json',{'config':cfg,'versions':versions,'difflogic_commit':UPSTREAM_COMMIT,
        'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data else None,
        'split_ids':{s:d['ids'][idx].tolist() for s,idx in splits.items()}})
    torch.save({'model_kwargs':kwargs,'state_dict':model.state_dict(),
                'oracle_concepts':args.oracle_concepts,'precomputed_concepts':args.precomputed_concepts,
                'threshold':threshold},args.out/'model.pt')
    np.savez_compressed(args.out/'preprocessing.npz',mean=mean,std=std,concept_names=d['concept_names'])
    np.savez_compressed(args.out/'calibration.npz',scores=qc,concepts=d['c'][splits['calibration']])
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__': main()
