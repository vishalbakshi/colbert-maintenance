## `colbertt-ai` 0.2.22 Release Testing Scripts

This folder contains testing scripts for the 0.2.22 release. 

### Dockerfiles

- Dockerfile.main: installs the stanford-futuredata/ColBERT repo's main branch (as of August 10, 2025) with latest torch (`2.8.0`) and transformers (`4.55.0`).
- Dockerfile.0.2.21.pypi.torch.1.13.1: installs `colbert-ai==0.2.21` from PyPI with `torch==1.31.1` and `transformers==4.38.2`.
- Dockerfile.0.2.22.testpypi.torch.1.13.1: installs the test build of the main branch pushed to TestPyPI with `torch==1.31.1` and `transformers==4.38.2`. This install is used to measure backwards compatibility with Dockerfile.0.2.21.
- Dockerfile.0.2.22.testpypi.torch.2.8.0: installs the test build of the main branch pushed to TestPyPI with `torch==2.8.0` and `transformers==4.55.0`. This install is used to compare with the `main` branch install as a "pre-release" test.
- Dockerfile.0.2.22.pypi: installs `colbert-ai==0.2.22` from PyPI with `torch==2.8.0` and `transformers==4.55.0` (the latest versions of each library as of August 10, 2025).
- Dockerfile.compare: installs torch, ranx and pandas for index and search artifact comparison.

## Indexing, Search and Training Code

[`main.py`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/main.py) consists of an `_index`, `_search` and `_train` function that are commented out according to what needs to be run. The following Modal command is used:

```
SOURCE="0.2.22.testpypi.torch.2.8.0" DATE="20250806" PROJECT="deprecated-AdamW" NRANKS=4 GPU="L4:4" MAXSTEPS=1000 modal run main.py
```

This will create a `20250806-0.2.22.testpypi.torch.2.8.0-4` folder in the `colbert-maintenance` Modal Volume and will create `indexing` and `search` subfolders within with their respective artifacts (index files, raw rankings) using Dockerfile.0.2.22.testpypi.torch.2.8.0.

## Index and Search Comparison

[`compare.py`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/compare.py) contains the following functions:

- `_data` loads and prints out the shapes of the TSVs used during training.
- `_compare_index` compares index filenames, tensor shapes and tensor value (with `torch.allclose`).
- `_compare_search` compares aggregate and query-level metrics and calculates the average number of different retrieved passages

Use the following Modal command to compare two sets of index and search artifacts:

```
PATH_A="deprecated-AdamW/20250806-pypi-1" PATH_B="deprecated-AdamW/20250806-testpypi-1" modal run compare.py
```

## Data Prep

Since `colbert-ai` expects training triples (qid, pos pid, neg pid) as a JSON file, [`prep_data.ipynb`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/prep_data.ipynb) converts the MSMARCO triples tsv to a JSON format.

## Training Plots

Training logs are copy-pasted from the terminal into `*_train_log.txt` files. You can plot two logs using [`Training_Plots.ipynb`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/Training_Plots.ipynb). 

The training logs have format:

```
#>>>    <pos doc scor> <neg doc scor>            |                <pos score - neg score>
[<timestamp>] <step #> <training loss>
```