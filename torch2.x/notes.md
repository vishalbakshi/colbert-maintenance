## torch==1.13.1 --> torch==2.0

I ran indexing (ConditionalQA), search (ConditionalQA) and training (MSMARCO) for the following two `colbert-ai` installs:

- A: `torch==1.13.1` + `transformers==4.38.2`
- B: `torch==2.0` + `transformers==4.38.2`

All index/search/training results are the same.

### Indexing Results

```
╭──────────────────────────────╮
│ COMPARING TENSOR DIRECTORIES │
╰──────────────────────────────╯
Path A: 
/colbert-maintenance/torch2.x/20250813-0.2.22.torch.1.13.1-1/indexing/Conditiona
lQA/
Path B: 
/colbert-maintenance/torch2.x/20250813-0.2.22.torch.2.0-1/indexing/ConditionalQA
/

INDEX FILE NAME COMPARISON
✓ All 18 files match

TENSOR SHAPE COMPARISON

0.codes.pt
  Shape: torch.Size([408459]) vs torch.Size([408459])

0.residuals.pt
  Shape: torch.Size([408459, 48]) vs torch.Size([408459, 48])

1.codes.pt
  Shape: torch.Size([416038]) vs torch.Size([416038])

1.residuals.pt
  Shape: torch.Size([416038, 48]) vs torch.Size([416038, 48])

2.codes.pt
  Shape: torch.Size([322440]) vs torch.Size([322440])

2.residuals.pt
  Shape: torch.Size([322440, 48]) vs torch.Size([322440, 48])

avg_residual.pt
  Shape: torch.Size([]) vs torch.Size([])

buckets.pt
  Tensor[0]: torch.Size([15]) vs torch.Size([15])
  Tensor[1]: torch.Size([16]) vs torch.Size([16])

centroids.pt
  Shape: torch.Size([16384, 96]) vs torch.Size([16384, 96])

ivf.pid.pt
  Tensor[0]: torch.Size([975337]) vs torch.Size([975337])
  Tensor[1]: torch.Size([16384]) vs torch.Size([16384])

TENSOR VALUE COMPARISON

0.codes.pt
  ✓ Values match

0.residuals.pt
  ✓ Values match

1.codes.pt
  ✓ Values match

1.residuals.pt
  ✓ Values match

2.codes.pt
  ✓ Values match

2.residuals.pt
  ✓ Values match

avg_residual.pt
  ✓ Values match

buckets.pt
  ✓ Tensor[0] values match
  ✓ Tensor[1] values match

centroids.pt
  ✓ Values match

ivf.pid.pt
  ✓ Tensor[0] values match
  ✓ Tensor[1] values match
╭───────── FINAL SUMMARY ──────────╮
│ Files processed: 10              │
│ Shape matches: 10/10             │
│ Value matches: 10/10             │
│                                  │
│ 🎉 ALL TENSORS ARE IDENTICAL! 🎉 │
╰──────────────────────────────────╯
```

### Search Results

```
╭─────────────────────────────╮
│ COMPARING RETRIEVAL RESULTS │
╰─────────────────────────────╯
Path A: 
/colbert-maintenance/torch2.x/20250813-0.2.22.torch.1.13.1-1/search/ConditionalQ
A.tsv
Path B: 
/colbert-maintenance/torch2.x/20250813-0.2.22.torch.2.0-1/search/ConditionalQA.t
sv

METRICS COMPARISON
/opt/conda/envs/compare/lib/python3.11/site-packages/ranx/metrics/recall.py:29: NumbaTypeSafetyWarning: unsafe cast from uint64 to int64. Precision may be lost.
  scores[i] = _recall(qrels[i], run[i], k, rel_lvl)
✓ Mean Recall@10 Matches
        A: 0.1309418985666801
        B: 0.1309418985666801
✓ Mean MRR@10 Matches
        A: 0.1769138405669771
        B: 0.1769138405669771

AVG NUMBER OF DIFFERENT PASSAGES RETRIEVED
0.0

QUERY-LEVEL RECALL@10 DIFFERENCE
271
Equal    271
Name: count, dtype: int64

QUERY-LEVEL MRR@10 DIFFERENCE
271
Equal    271
Name: count, dtype: int64

```

### Training Results

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/11941df2-00b9-475f-90d2-416d4955c420" />

## torch==1.13.1 --> torch==2.0.1

This yields identical index, search and training artifacts.

```
╭───────── FINAL SUMMARY ──────────╮
│ Files processed: 10              │
│ Shape matches: 10/10             │
│ Value matches: 10/10             │
│                                  │
│ 🎉 ALL TENSORS ARE IDENTICAL! 🎉 │
╰──────────────────────────────────╯
```

```
✓ Mean Recall@10 Matches
        A: 0.1309418985666801
        B: 0.1309418985666801
✓ Mean MRR@10 Matches
        A: 0.1769138405669771
        B: 0.1769138405669771

AVG NUMBER OF DIFFERENT PASSAGES RETRIEVED
0.0

QUERY-LEVEL RECALL@10 DIFFERENCE
271
Equal    271
Name: count, dtype: int64

QUERY-LEVEL MRR@10 DIFFERENCE
271
Equal    271
Name: count, dtype: int64
```

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/309065e5-80be-429f-b4fb-98f76592ac72" />


## torch==1.13.1 --> torch==2.1

This introduces changes in index, search and training artifacts.

### Indexing Results

Only highlighting differences

```
ivf.pid.pt
  Tensor[0]: torch.Size([975337]) vs torch.Size([973110])

TENSOR VALUE COMPARISON

0.codes.pt
  ✗ Values differ

0.residuals.pt
  ✗ Values differ

1.codes.pt
  ✗ Values differ

1.residuals.pt
  ✗ Values differ

2.codes.pt
  ✗ Values differ

2.residuals.pt
  ✗ Values differ

avg_residual.pt
  ✗ Values differ

buckets.pt
  ✗ Tensor[0] values differ
  ✗ Tensor[1] values differ

centroids.pt
  ✗ Values differ

ivf.pid.pt
  ✗ Tensor[0] shape mismatch
  ✗ Tensor[1] values differ
```

### Search Results

```
╭─────────────────────────────╮
│ COMPARING RETRIEVAL RESULTS │
╰─────────────────────────────╯
Path A: 
/colbert-maintenance/torch2.x/20250813-0.2.22.torch.1.13.1-1/search/ConditionalQ
A.tsv
Path B: 
/colbert-maintenance/torch2.x/20250813-0.2.22.torch.2.1-1/search/ConditionalQA.t
sv

METRICS COMPARISON
/opt/conda/envs/compare/lib/python3.11/site-packages/ranx/metrics/recall.py:29: NumbaTypeSafetyWarning: unsafe cast from uint64 to int64. Precision may be lost.
  scores[i] = _recall(qrels[i], run[i], k, rel_lvl)
✗ Mean Recall@10 Doesn't Match
        A: 0.1309418985666801
        B: 0.13217191086680308
✗ Mean MRR@10 Doesn't Match
        A: 0.1769138405669771
        B: 0.18163913781994961

AVG NUMBER OF DIFFERENT PASSAGES RETRIEVED
1.2250922509225093

QUERY-LEVEL RECALL@10 DIFFERENCE
271
Equal       268
Increase      2
Decrease      1
Name: count, dtype: int64

QUERY-LEVEL MRR@10 DIFFERENCE
271
Equal       254
Increase     11
Decrease      6
Name: count, dtype: int64
```
