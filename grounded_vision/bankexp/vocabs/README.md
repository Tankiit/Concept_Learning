# Concept vocabularies

`cub_bound_28.txt` — the 28 bound CUB attributes (part, attribute,
value) used by the bank, one per line as `part:attribute=value`,
generated from `intermediate_results/meta.json` (which also carries the
8 structural `has_part(part)` features; total bank width 36).

Regenerate after refitting the bank:

    python -c "import json;print('\n'.join(f'{p}:{a}={v}' for p,a,v in json.load(open('../intermediate_results/meta.json'))['vocab']))" > cub_bound_28.txt
