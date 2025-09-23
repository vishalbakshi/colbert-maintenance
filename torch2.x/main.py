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
NRANKS = int(os.environ.get("NRANKS", 1))
ROOT = os.environ.get("ROOT", "")
NDOCS = int(os.environ.get("NDOCS", 70000))
SWAP = os.environ.get("SWAP", "")
SWAP_ROOT = os.environ.get("SWAP_ROOT", "")
POSTNORM_CENTROIDS_SWAP = os.environ.get("POSTNORM_CENTROIDS_SWAP", "")
POSTNORM_CENTROIDS_SWAP_ROOT = os.environ.get("POSTNORM_CENTROIDS_SWAP_ROOT", "")
ROOT_A = os.environ.get("ROOT_A", "")
ROOT_B = os.environ.get("ROOT_B", "")
image = Image.from_dockerfile(f"Dockerfile.{SOURCE}", gpu="L4")

if ROOT != "":
    image = image.add_local_file("collection_indexer.py", "/ColBERT/colbert/indexing/collection_indexer.py")
    image = image.add_local_file("checkpoint.py", "/ColBERT/colbert/modeling/checkpoint.py")
    image = image.add_local_file("index_saver.py", "/ColBERT/colbert/indexing/index_saver.py")
    image = image.add_local_file("residual.py", "/ColBERT/colbert/indexing/codecs/residual.py")

app = App(PROJECT)

print(f"Source: {SOURCE}")
print(f"Project: {PROJECT}")
print(f"Date: {DATE}")
print(f"GPU: {GPU}")
print(f"NRANKS: {NRANKS}")
print(f"NDOCS: {NDOCS}")
print(f"SWAP: {SWAP}")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _swap_search(root_a, root_b, nranks):
    import torch
    print(torch.__version__)
    from colbert.data import Queries
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher
    from datasets import load_dataset

    results = {}

    dataset_name = "ConditionalQA"
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")

    queries_dict = {}
    for item in queries: queries_dict[item['_id']] = item['text']

    with Run().context(RunConfig(nranks=nranks)):
        config = ColBERTConfig(
                ncells=4,
                centroid_score_threshold=0.45,
                ndocs=1024,
            )

        searcher = Searcher(index=dataset_name, index_root=f"{MOUNT}/{root_a}/indexing", config=config)
        ranking = searcher.search_all(queries_dict, k=10)
        ranking.save(f"{MOUNT}/{root_b}/search/{dataset_name}_swap.tsv")

    print("Swap search complete!")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _subtract(source, project, date, nranks):
    import torch
    b = torch.randn((153894, 96), generator=torch.Generator().manual_seed(42)).half()
    c = torch.randn((153894, 96), generator=torch.Generator().manual_seed(13)).half()
    r = b - c
    torch.save(r, f"{MOUNT}/{project}/{date}-{source}-{nranks}/subtract_r.pt")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _sort(source, project, date, nranks):
    import torch
    torch.manual_seed(42) 
    t = torch.randint(low=0, high=16383, size=(1146937,))
    torch.save(t, f"{MOUNT}/{project}/{date}-{source}-{nranks}/randint_t.pt")
    t = t.sort()
    torch.save(t, f"{MOUNT}/{project}/{date}-{source}-{nranks}/randint_t_sorted.pt")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _lse(source, project, date, nranks, ndocs=70000):
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    from datasets import load_dataset

    print(torch.__version__)
    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    print("passages loaded!")

    config = ColBERTConfig(
        doc_maxlen=256,      
        nbits=4,             
        dim=96,             
        kmeans_niters=20,
        index_bsize=32,
        bsize=64,
        checkpoint="answerdotai/answerai-colbert-small-v1"
    )

    checkpoint = Checkpoint("answerdotai/answerai-colbert-small-v1", colbert_config=config)
    torch.save(checkpoint.bert.state_dict(), f'{MOUNT}/{project}/{date}-{source}-{nranks}/bert_weights.pt')
    print("checkpoint.bert.state_dict saved!")

    sample_pids = torch.load(f"{MOUNT}/{project}/{date}-{source}-{nranks}/sample_pids.pt")

    idx = 0
    batch_idx = 68
    for idx in range(29):
    # for idx in [2, 3, 4, 5, 12, 15, 18, 19, 20, 26]:
        docs = passages['text'][list(sample_pids)[1600*idx:1600*(idx+1)]]
        text_batches, reverse_indices = checkpoint.doc_tokenizer.tensorize(docs, bsize=config.index_bsize)
        input_ids = text_batches[0][0][:, :batch_idx]
        attention_mask = text_batches[0][1][:, :batch_idx]
        # print(idx, len(docs), len(sample_pids), input_ids.shape)
        if idx == 2: print(idx, input_ids.shape)

        with torch.cuda.amp.autocast():
            outputs_dict = {}
            def capture_output(name):
                def hook_fn(module, input, output):
                    outputs_dict[name] = output[0].detach()
                return hook_fn

            hooks = []
            for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
            with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
            for h in hooks: h.remove()

            torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/lse_outputs_dict_{idx}.pt")
            print(f"lse_outputs_dict_{idx} saved!")
    
@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _norm(source, project, date, nranks):
    import torch
    print(torch.__version__)

    torch.manual_seed(13)
    t = torch.empty(1024, 96).uniform_(-0.4, 0.4)
    torch.save(t, f"{MOUNT}/{project}/{date}-{source}-{nranks}/t.pt")
    torch.save(t.half(), f"{MOUNT}/{project}/{date}-{source}-{nranks}/half_t.pt")

    t = torch.nn.functional.normalize(t, dim=-1)
    torch.save(t, f"{MOUNT}/{project}/{date}-{source}-{nranks}/norm.pt")
    torch.save(t.half(), f"{MOUNT}/{project}/{date}-{source}-{nranks}/half_norm.pt")
    print("Normalized random tensor saved!")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _remove_dense(source, project, date, nranks):
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    print(torch.__version__)

    config = ColBERTConfig(
        doc_maxlen=256,      
        nbits=4,             
        dim=96,             
        kmeans_niters=20,
        index_bsize=32,
        bsize=64,
        checkpoint="answerdotai/answerai-colbert-small-v1"
    )

    checkpoint = Checkpoint("answerdotai/answerai-colbert-small-v1", colbert_config=config)

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    for layer in checkpoint.bert.encoder.layer:
        layer.intermediate.dense = Identity()
        layer.output.dense = Identity()

    text_batches, reverse_indices = torch.load(f'{MOUNT}/{project}/{date}-{source}-{nranks}/tensorize_output.pt')
    input_ids = text_batches[0][0][:8] # 8 items of first batch
    attention_mask = text_batches[0][1][:8] # 8 items of first batch
    print(input_ids.shape)

    outputs_dict = {}
    def capture_output(name):
        def hook_fn(module, input, output):
            outputs_dict[name] = output[0].detach()
        return hook_fn

    with torch.cuda.amp.autocast():
        hooks = []
        for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
        with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
        for h in hooks: h.remove()
        torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/rd_outputs_dict.pt")
        print("rd_outputs_dict saved!")

        D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
        torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_rd_bert.pt")
        print("D_rd_bert.pt saved!")

        D = checkpoint.linear(D)
        torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_rd_linear.pt")
        print("D_rd_linear.pt saved!")

        mask = torch.tensor(checkpoint.mask(input_ids, skiplist=checkpoint.skiplist), device=checkpoint.device).unsqueeze(2).float()
        D = D * mask
        torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_rd_mask.pt")
        print("D_rd_mask.pt saved!")

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_rd_norm.pt")
        print("D_rd_norm.pt saved!")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _amp_bert(source, project, date, nranks):
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    from datasets import load_dataset

    print(torch.__version__)
    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")

    config = ColBERTConfig(
        doc_maxlen=256,      
        nbits=4,             
        dim=96,             
        kmeans_niters=20,
        index_bsize=32,
        bsize=64,
        checkpoint="answerdotai/answerai-colbert-small-v1"
    )

    checkpoint = Checkpoint("answerdotai/answerai-colbert-small-v1", colbert_config=config)
    torch.save(checkpoint.bert.state_dict(), f'{MOUNT}/{project}/{date}-{source}-{nranks}/bert_weights.pt')
    print("checkpoint.bert.state_dict saved!")

    docs = ["a"]
    # sample_pids = torch.load(f"{MOUNT}/{project}/{date}-{source}-{nranks}/sample_pids.pt")
    # docs = passages['text'][list(sample_pids)[1600*idx:1600*(idx+1)]]
    # docs = passages['text'][:1000]
    text_batches, reverse_indices = checkpoint.doc_tokenizer.tensorize(docs, bsize=config.index_bsize)
    input_ids = text_batches[0][0] 
    attention_mask = text_batches[0][1] 

    outputs_dict = {}
    def capture_output(name):
        def hook_fn(module, input, output):
            outputs_dict[name] = output[0].detach()
        return hook_fn

    with torch.cuda.amp.autocast():
        hooks = []
        for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
        with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
        for h in hooks: h.remove()
        torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/amp_outputs_dict.pt")
        print("amp_outputs_dict saved!")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _layer_norm():
    import torch
    print(torch.__version__)

    torch.manual_seed(42)
    layernorm = torch.nn.LayerNorm(96).cuda()
    layernorm.eval()

    with torch.no_grad():
        x32 = torch.randn(32, 71, 96).cuda()
        x8 = x32[:8]
        
        with torch.cuda.amp.autocast():
            out32 = layernorm(x32)
            out8 = layernorm(x8)
        
        print(f"Batch32 mean: {out32.mean().item():.10f}")
        print(f"Batch8 mean: {out8.mean().item():.10f}")
        print(f"Match first 8: {torch.allclose(out32[:8], out8)}")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _bert(source, project, date, nranks):
    import torch
    from colbert.infra import ColBERTConfig
    from colbert.modeling.checkpoint import Checkpoint
    print(torch.__version__)

    config = ColBERTConfig(
        doc_maxlen=256,      
        nbits=4,             
        dim=96,             
        kmeans_niters=20,
        index_bsize=32,
        bsize=64,
        checkpoint="answerdotai/answerai-colbert-small-v1"
    )

    checkpoint = Checkpoint("answerdotai/answerai-colbert-small-v1", colbert_config=config)
    torch.save(checkpoint.bert.state_dict(), f'{MOUNT}/{project}/{date}-{source}-{nranks}/bert_weights.pt')
    print("checkpoint.bert.state_dict saved!")

    docs = ['your immigration status changes, if you’re not a British citizen']
    text_batches, reverse_indices = checkpoint.doc_tokenizer.tensorize(docs, bsize=config.index_bsize)
    # text_batches, reverse_indices = torch.load(f'{MOUNT}/{project}/{date}-{source}-{nranks}/tensorize_output.pt')
    input_ids = text_batches[0][0]
    attention_mask = text_batches[0][1]
    
    outputs_dict = {}
    def capture_output(name):
        def hook_fn(module, input, output):
            outputs_dict[name] = output[0].detach()
        return hook_fn

    hooks = []
    for i in range(12): hooks.append(checkpoint.bert.encoder.layer[i].register_forward_hook(capture_output(f"{i}")))
    with torch.no_grad(): D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
    for h in hooks: h.remove()
    torch.save(outputs_dict, f"{MOUNT}/{project}/{date}-{source}-{nranks}/outputs_dict.pt")
    print("outputs_dict saved!")

    D = checkpoint.bert(input_ids, attention_mask=attention_mask)[0]
    torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_bert.pt")
    print("D_bert.pt saved!")

    D = checkpoint.linear(D)
    torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_linear.pt")
    print("D_linear.pt saved!")

    mask = torch.tensor(checkpoint.mask(input_ids, skiplist=checkpoint.skiplist), device=checkpoint.device).unsqueeze(2).float()
    D = D * mask
    torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_mask.pt")
    print("D_mask.pt saved!")

    D = torch.nn.functional.normalize(D, p=2, dim=2)
    torch.save(D, f"{MOUNT}/{project}/{date}-{source}-{nranks}/D_norm.pt")
    print("D_norm.pt saved!")


@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _index(source, project, date, nranks, ndocs, root, swap, swap_root, postnorm_centroids_swap, postnorm_centroids_swap_root):
    import os
    import subprocess
    subprocess.run(['pwd'], text=True, shell=True)
    from colbert import Indexer
    from colbert.infra import RunConfig, ColBERTConfig
    from colbert.infra.run import Run
    from datasets import load_dataset

    os.environ["ROOT"] = root
    os.environ["SWAP"] = swap
    os.environ["SWAP_ROOT"] = swap_root
    os.environ["POSTNORM_CENTROIDS_SWAP"] = postnorm_centroids_swap
    os.environ["POSTNORM_CENTROIDS_SWAP_ROOT"] = postnorm_centroids_swap_root

    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")
    qrels_rows = load_dataset("UKPLab/dapr", f"{dataset_name}-qrels", split="test")

    with Run().context(RunConfig(nranks=nranks)):
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
        _ = indexer.index(name=f"{MOUNT}/{project}/{date}-{source}-{nranks}/indexing/{dataset_name}", collection=passages[:ndocs]["text"], overwrite=True)

    print("Index created!")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _search(source, project, date, nranks):
    from colbert.data import Queries
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Searcher
    from datasets import load_dataset

    results = {}

    dataset_name = "ConditionalQA"
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")

    queries_dict = {}
    for item in queries: queries_dict[item['_id']] = item['text']

    with Run().context(RunConfig(nranks=nranks)):
        config = ColBERTConfig(
                ncells=4,
                centroid_score_threshold=0.45,
                ndocs=1024,
            )

        searcher = Searcher(index=dataset_name, index_root=f"{MOUNT}/{project}/{date}-{source}-{nranks}/indexing", config=config)
        ranking = searcher.search_all(queries_dict, k=10)
        ranking.save(f"{MOUNT}/{project}/{date}-{source}-{nranks}/search/{dataset_name}.tsv")

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
def _train(source, project, date, maxsteps, nranks):
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert import Trainer

    with Run().context(RunConfig(nranks=nranks)):

        config = ColBERTConfig(
            bsize=32,
            root=f"{MOUNT}/{project}/{date}-{source}-{nranks}/training/",
            maxsteps=maxsteps
        )

        trainer = Trainer(
            triples=f"{MOUNT}/data/triples.train.small.json",
            queries=f"{MOUNT}/data/queries.train.tsv",
            collection=f"{MOUNT}/data/collection.tsv",
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _versions():
    import torch
    import transformers
    import subprocess
    import datasets
    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    _ = subprocess.run([PYTHON_PATH, "-m", "pip", "show", "colbert-ai"], text=True)
    print(f"datasets: {datasets.__version__}")

@app.local_entrypoint()
def main():
    import time
    # _versions.remote()
    # _index.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS, ndocs=NDOCS, root=ROOT, swap=SWAP, swap_root=SWAP_ROOT, postnorm_centroids_swap=POSTNORM_CENTROIDS_SWAP, postnorm_centroids_swap_root=POSTNORM_CENTROIDS_SWAP_ROOT)
    # time.sleep(5)
    # _search.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    # _data.remote()
    # time.sleep(5)
    # _train.remote(source=SOURCE, project=PROJECT, date=DATE, maxsteps=MAXSTEPS, nranks=NRANKS)
    # _bert.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    # _layer_norm.remote()
    # _amp_bert.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    # _remove_dense.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    # _norm.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    # _lse.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS, ndocs=NDOCS)
    # _sort.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    # _subtract.remote(source=SOURCE, project=PROJECT, date=DATE, nranks=NRANKS)
    _swap_search.remote(root_a=ROOT_A, root_b=ROOT_B, nranks=NRANKS)