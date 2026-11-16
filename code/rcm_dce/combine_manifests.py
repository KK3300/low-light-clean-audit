"""Combine explicitly paired manifests with dataset namespaces."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

def read(p):
    with Path(p).open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',nargs='+',required=True,help='name=manifest.csv');ap.add_argument('--output',required=True);args=ap.parse_args()
    rows=[]
    for item in args.spec:
        name,path=item.split('=',1)
        for r in read(path):
            q=dict(r);q['id']=f'{name}:{r.get("id","")}';q['scene_key']=f'{name}:{r.get("scene_key") or r.get("id","")}';q['source_dataset']=name;q['split']='combined';rows.append(q)
    fields=['input','target','id','scene_key','source_dataset','split'];out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report={'output':str(out.resolve()),'rows':len(rows),'datasets':{name:sum(1 for r in rows if r['source_dataset']==name) for name,_ in (x.split('=',1) for x in args.spec)},'unique_scene_keys':len({r['scene_key'] for r in rows})}
    out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
