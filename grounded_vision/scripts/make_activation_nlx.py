"""Faithful template NLX for certified, spatially grounded activation reasons."""
import json
from pathlib import Path

run = Path('/Users/cril/tanmoy/research/data/grounded_vision_runs/cub_bound_real_full')
source = run / 'activation_tree_grounded_explanations.jsonl'
out = run / 'activation_grounded_nlx.jsonl'

def phrase(g):
    c = g['condition']
    t = g['tile']
    return (f"DINO activation channel {g['activation_channel']} in spatial tile "
            f"(row {t['row'] + 1}, column {t['col'] + 1}) is {c['operator']} "
            f"{c['threshold']:.3f} (observed {c['observed_value']:.3f})")

with source.open() as fin, out.open('w') as fout:
    for line in fin:
        r = json.loads(line)
        clauses = [phrase(g) for g in r['grounding']]
        evidence = '; and '.join(clauses)
        r['nlx'] = {
            'text': f"The activation tree predicts class {r['prediction']} because {evidence}.",
            'style': 'activation_grounded_template',
            'faithfulness': 'Every clause is a certified tree condition with a spatial evidence tile.',
            'semantic_abstention': True,
        }
        fout.write(json.dumps(r) + '\n')
print(out)
