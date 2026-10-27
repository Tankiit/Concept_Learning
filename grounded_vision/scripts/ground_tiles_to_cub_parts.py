"""Attach validated CUB part-point semantics to activation-tile evidence."""
import json
from pathlib import Path
from PIL import Image

root=Path('/Users/cril/tanmoy/research/data/CUB_200_2011/CUB_200_2011')
run=Path('/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full')
names={int(x.split()[0]):x.split(maxsplit=1)[1] for x in (root/'parts/parts.txt').read_text().splitlines()}
images={int(x.split()[0]):x.split(maxsplit=1)[1] for x in (root/'images.txt').read_text().splitlines()}
points={}
for line in (root/'parts/part_locs.txt').read_text().splitlines():
    sid,pid,x,y,v=line.split()
    if int(v): points.setdefault(int(sid),[]).append((names[int(pid)],float(x),float(y)))

src=run/'activation_grounded_nlx.jsonl'; dst=run/'activation_part_grounded_nlx.jsonl'
with src.open() as fin, dst.open('w') as fout:
  for line in fin:
    r=json.loads(line); im=Image.open(root/'images'/images[r['sample_id']]); w,h=im.size
    clauses=[]
    for g in r['grounding']:
      b=g['tile']['image_box_normalized']; parts=[n for n,x,y in points.get(r['sample_id'],[]) if b[0]<=x/w<=b[2] and b[1]<=y/h<=b[3]]
      g['validated_cub_parts']=parts
      where=('the '+', '.join(parts)+' area' if parts else 'an unannotated image region')
      c=g['condition']; clauses.append(f"activation channel {g['activation_channel']} in {where} is {c['operator']} {c['threshold']:.3f}")
    r['nlx']['text']=f"The activation tree predicts class {r['prediction']} because " + '; and '.join(clauses) + '.'
    r['nlx']['semantic_grounding']='CUB visible part points contained in the evidence tile; no unvalidated attribute claims.'
    part_phrases=list(dict.fromkeys([('the '+', '.join(g['validated_cub_parts'])+' area') if g['validated_cub_parts'] else 'an unannotated image region' for g in r['grounding']]))
    r['nlx_part_only']={
      'text':f"The activation tree predicts class {r['prediction']} from verified evidence in " + ', and '.join(part_phrases) + '.',
      'faithfulness':'Part phrases refer only to tiles used by verified conditions; the full literals remain in certificate.conditions.',
      'semantic_abstention':True,
    }
    fout.write(json.dumps(r)+'\n')
print(dst)
