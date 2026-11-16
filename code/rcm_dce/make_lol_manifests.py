"""Build explicit LOL-v1 train/validation/test manifests from the audited split."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def write(path, rows, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['input','target','id','scene_key','split']); w.writeheader()
        for r in rows:
            w.writerow({**r,'split':split})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='experiments/datasets/LOL_v1'); ap.add_argument('--output-dir',default='experiments/datasets/LOL_v1/rcm_manifests'); args=ap.parse_args()
    root=Path(args.root).resolve(); out=Path(args.output_dir).resolve(); split=json.loads((root/'scene_split_80_20/split.json').read_text(encoding='utf8'))
    def rows_for(ids, base):
        rows=[]
        for sid in ids:
            low=base/'low'/f'{sid}.png'; high=base/'high'/f'{sid}.png'
            if not low.exists() or not high.exists(): raise FileNotFoundError((low,high))
            rows.append({'input':str(low),'target':str(high),'id':str(sid),'scene_key':str(sid)})
        return rows
    train=rows_for(split['train_ids'],root/'extracted/our485'); val=rows_for(split['val_ids'],root/'extracted/our485')
    test_ids=sorted(p.stem for p in (root/'extracted/eval15/low').glob('*.png'))
    test=rows_for(test_ids,root/'extracted/eval15')
    write(out/'train_pairs.csv',train,'scene_disjoint_train'); write(out/'val_pairs.csv',val,'scene_disjoint_val'); write(out/'test_pairs.csv',test,'official_eval15')
    report={'root':str(root),'output_dir':str(out),'train_rows':len(train),'val_rows':len(val),'test_rows':len(test),'train_val_overlap':sorted(set(x['id'] for x in train)&set(x['id'] for x in val)),'train_test_overlap':sorted(set(x['id'] for x in train)&set(x['id'] for x in test)),'val_test_overlap':sorted(set(x['id'] for x in val)&set(x['id'] for x in test))}
    (out/'manifest_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
