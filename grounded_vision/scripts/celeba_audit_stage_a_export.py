"""Stage A: flwrlabs/celeba parquet -> jpg files + label npz per split.

Usage: python stage_a_export.py <split> <parquet...>
Writes: imgs_<split>/img_%06d.jpg, <split>_attrs.npy (n,40 bool), <split>_ids.npy
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

RUN = Path(__file__).parent


def export(split: str, paths):
    out_dir = RUN / f"imgs_{split}"
    out_dir.mkdir(exist_ok=True)
    attrs_path = RUN / f"{split}_attrs.npy"
    ids_path = RUN / f"{split}_ids.npy"
    if attrs_path.exists():
        print(f"[skip] {split} already exported")
        return
    cols = None
    n = 0
    buf_a, buf_i = [], []
    for p in map(Path, paths):
        f = pq.ParquetFile(p)
        if cols is None:
            cols = [c for c in f.schema_arrow.names if c not in ("image", "celeb_id")]
            print(f"{split}: {len(cols)} attribute columns")
        for rg in range(f.num_row_groups):
            t = f.read_row_group(rg)
            imgs = t.column("image").to_pylist()
            ids = t.column("celeb_id").to_pylist()
            attr_rows = list(zip(*[t.column(c).to_pylist() for c in cols]))
            for img, cid, arow in zip(imgs, ids, attr_rows):
                (out_dir / f"img_{n:06d}.jpg").write_bytes(img["bytes"])
                buf_a.append(arow)
                buf_i.append(cid)
                n += 1
        print(f"  {p.name}: total {n}")
    np.save(attrs_path, np.array(buf_a, dtype=bool))
    np.save(ids_path, np.array(buf_i, dtype=np.int64))
    (RUN / f"{split}_columns.txt").write_text("\n".join(cols))
    print(f"[done] {split}: {n} images, attrs {np.array(buf_a).shape}")


if __name__ == "__main__":
    split = sys.argv[1]
    export(split, sys.argv[2:])
