## `colbertt-ai` 0.2.22 Release Testing Scripts

This folder contains testing scripts for the 0.2.22 release. 

### Dockerfiles

- [Dockerfile.main](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/Dockerfile.main): installs the GitHub repo's main branch + latest torch and transformers.
- [Dockerfile.pypi](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/Dockerfile.pypi): installs 0.2.21 on PyPI.
- [Dockerfile.testpypi](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/Dockerfile.testpypi): installs the build of the main branch pushed to PyPI.
- Dockerfile.newpypi: installs 0.2.22 from PyPI.
- [Dockerfile.compare](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/Dockerfile.compare): installs torch, ranx and pandas for index and search artifact comparison.

## Indexing, Search and Training Code

[`main.py`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/main.py) consists of an `_index`, `_search` and `_train` function that are commented out according to what needs to be run. The following Modal command is used:

```
SOURCE="pypi" DATE="20250806" PROJECT="deprecated-AdamW" NRANKS=4 GPU="L4:4" MAXSTEPS=1000 modal run main.py
```

This will create a `20250806-pypi-4` folder in the `colbert-maintenance` Modal Volume and will create `indexing` and `search` subfolders within with their respective artifacts (index files, raw rankings).

## Index and Search Comparison

[`compare.py`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/compare.py) contains the following functions:

- `_data` loads and prints out the shapes of the TSVs used during training.
- `_compare_index` compares index filenames, tensor shapes and tensor value (with `torch.allclose`).
- `_compare_search` compares aggregate and query-level metrics and calculates the average number of different retrieved passages

Use the following Modal command:

```
PATH_A="deprecated-AdamW/20250806-pypi-1" PATH_B="deprecated-AdamW/20250806-testpypi-1" modal run compare.py
```

## Data Prep

Since `colbert-ai` expects training triples (qid, pos pid, neg pid) as a JSON file, [`prep_data.ipynb`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/prep_data.ipynb) converts the MSMARCO triples tsv to a JSON format.

## Training Plots

Training logs are copy-pasted from the terminal into `*_train_log.txt` files. You can plot two logs using [`Training_Plots.ipynb`](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/Training_Plots.ipynb). Plots of interest:

- [1.13.1-vs-2.7.1.png](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/1.13.1-vs-2.7.1.png): compares 0.2.21 PyPI and main branch `colbert-ai` installs for single-GPU training.
- [1.13.1-vs-2.7.1_4.png](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/1.13.1-vs-2.7.1_4.png): compares 0.2.21 PyPI and main branch `colbert-ai` installs for multi-GPU training.
- [main-vs-testpypi.png](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/main-vs-testpypi.png): compares 0.2.22 main branch and TestPyPI installs for single-GPU training.
- [main-vs-testpypi-4.png](https://github.com/vishalbakshi/colbert-maintenance/blob/main/0.2.22/main-vs-testpypi-4.png): compares 0.2.22 main branch and TestPyPI installs for multi-GPU training.

The training logs have format:

```
#>>>    <pos doc scor> <neg doc scor>            |                <pos score - neg score>
[<timestamp>] <step #> <training loss>
```