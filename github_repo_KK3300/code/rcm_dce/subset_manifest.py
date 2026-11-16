"""Create a deterministic, scene-disjoint audit subset from a manifest."""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path


def rank(row, seed):
    scene = row.get("scene_key") or row.get("id", "")
    return hashlib.sha256(f"{seed}:{scene}:{row.get('id','')}".encode()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);ap.add_argument('--count',type=int,required=True);ap.add_argument('--seed',type=int,default=20260830);args=ap.parse_args()
    src=Path(args.input).resolve();out=Path(args.output).resolve()
    with src.open('r',encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    rows=sorted(rows,key=lambda r:rank(r,args.seed))[:args.count]
    fields=list(rows[0].keys()) if rows else ['input','target','id','scene_key','split']
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report={'source':str(src),'output':str(out),'seed':args.seed,'requested':args.count,'rows':len(rows),'scene_keys':len({r.get('scene_key') for r in rows})}
    out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
