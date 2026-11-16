# Manifests

## `train_pairs.csv` — 8,040 rows

Clean combined training manifest: LSD-DLL 7,652 + LOL-v1 388.

Columns:

| Column | Meaning |
|---|---|
| `input` | Relative path to the low-light input |
| `target` | Relative path to the normal-light target |
| `id` | Pair identifier (`LSD:<file>` or `LOL:<file>`) |
| `scene_key` | Scene key used for split integrity |
| `source_dataset` | `LSD`, `LOL`, or `combined` |
| `split` | Manifest split label |

## `val_pairs.csv` — 479 rows

Clean combined validation manifest: LSD 382 + LOL 97.

## `../audit/identity_pairs.csv` — 434 rows

The byte-identical input–target pairs found in the 10,000-pair LSD-DLL mirror. Columns:

`split,id,scene_key,filename_group,size_bytes,sha256`

Counts by level (manuscript Section 4.2):

- corpus: 434 / 10,000 (4.34%)
- original training split: 355 / 8,007 (4.43%)
- original validation split: 79 / 1,993 (3.96%)
- combined training manifest: 355 / 8,395
- combined validation manifest: 18 / 497

## `../audit/content_leak_pairs.csv` — 12 rows

Additional content-leak pairs found during the audit; not part of the 434 byte-identical count.

## Paths

The `input`/`target` values are relative to a project root that contains:

```text
experiments/datasets/LSD_DLL_subset_10000/{input,gt}/
experiments/datasets/LOL_v1/...
```

No image data is redistributed. See the top-level `README.md` and `REPRODUCE.md`.
