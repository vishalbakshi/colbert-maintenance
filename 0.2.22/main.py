import modal
from modal import Image, App, Secret, Volume
import datetime
import os

SOURCE = os.environ.get("SOURCE", "")
DATE = os.environ.get("DATE", "")
PROJECT = os.environ.get("PROJECT", "")
GPU = os.environ.get("GPU", "L4") 
PYTHON_PATH = "/opt/conda/envs/colbert/bin/python"
VOLUME = Volume.from_name("colbert-maintenance", create_if_missing=True)
MOUNT = "/colbert-maintenance"
MAXSTEPS = int(os.environ.get("MAXSTEPS", 10))
image = Image.from_dockerfile(f"Dockerfile.{SOURCE}", gpu=GPU)
app = App(PROJECT)

print(f"Source: {SOURCE}")
print(f"Project: {PROJECT}")
print(f"Date: {DATE}")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _index(source, project, date):
    from colbert import Indexer
    from colbert.infra import RunConfig, ColBERTConfig
    from colbert.infra.run import Run
    from datasets import load_dataset

    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")
    qrels_rows = load_dataset("UKPLab/dapr", f"{dataset_name}-qrels", split="test")

    with Run().context(RunConfig(nranks=1)):
        config = ColBERTConfig(
            doc_maxlen=256,      
            nbits=4,             
            dim=96,             
            kmeans_niters=20,
            index_bsize=32,
            bsize=64,
            checkpoint="answerdotai/answerai-colbert-small-v1"
        )
        
        indexer = Indexer(checkpoint="answerdotai/answerai-colbert-small-v1", config=config)
        _ = indexer.index(name=f"{MOUNT}/{project}/{date}-{source}/indexing/{dataset_name}", collection=passages[:70000]["text"], overwrite=True)

    print("Index created!")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _search(source, project, date):
    from colbert.data import Queries
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher
    from datasets import load_dataset

    results = {}

    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")
    qrels_rows = load_dataset("UKPLab/dapr", f"{dataset_name}-qrels", split="test")

    queries_dict = {}
    for item in queries: queries_dict[item['_id']] = item['text']

    with Run().context(RunConfig(nranks=1)):

        config = ColBERTConfig(
                ncells=4,
                centroid_score_threshold=0.45,
                ndocs=1024,
            )

        searcher = Searcher(index=dataset_name, index_root=f"{MOUNT}/{project}/{date}-{source}/indexing", config=config)
        ranking = searcher.search_all(queries_dict, k=10)
        ranking.save(f"{MOUNT}/{project}/{date}-{source}/search/{dataset_name}.tsv")

    print("Search complete!")

@app.function(image=modal.Image.debian_slim(python_version="3.10").pip_install("pandas"), timeout=10800,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _data():
    import pandas as pd
    from urllib.request import urlretrieve
    import tarfile
    import os

    urls = [
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz",
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz", 
            "https://msmarco.z22.web.core.windows.net/msmarcoranking/triples.train.small.tar.gz"
        ]
        
    extract_path = f"{MOUNT}/{PROJECT}/data/"
    os.makedirs(extract_path, exist_ok=True)
    
    for url in urls:
        filename = url.split('/')[-1]
        urlretrieve(url, filename)
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(extract_path)
        os.remove(filename)

    files = [
        f"{extract_path}/collection.tsv",
        f"{extract_path}/queries.dev.tsv",
        f"{extract_path}/queries.eval.tsv",
        f"{extract_path}/queries.train.tsv",
        f"{extract_path}/triples.train.small.tsv",
    ]

    for f in files: print(f, len(pd.read_csv(f, sep='\t', header=None)))

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _train(source, project, date, maxsteps):
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Trainer

    with Run().context(RunConfig(nranks=1)):

        config = ColBERTConfig(
            bsize=32,
            root=f"{MOUNT}/{project}/{date}-{source}/training/",
            maxsteps=maxsteps
        )

        trainer = Trainer(
            triples=f"{MOUNT}/{project}/data/triples.train.small.json",
            queries=f"{MOUNT}/{project}/data/queries.train.tsv",
            collection=f"{MOUNT}/{project}/data/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")

@app.local_entrypoint()
def main():
    # import time
    #_index.remote(source=SOURCE, project=PROJECT, date=DATE)
    #time.sleep(5)
    #_search.remote(source=SOURCE, project=PROJECT, date=DATE)
    #_data.remote()
    _train.remote(source=SOURCE, project=PROJECT, date=DATE, maxsteps=MAXSTEPS, )