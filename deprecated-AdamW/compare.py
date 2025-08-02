import modal
from modal import Image, App, Secret, Volume
import datetime
import os

PATH_A = os.environ.get("PATH_A", "")
PATH_B = os.environ.get("PATH_B", "")
GPU = os.environ.get("GPU", "L4") 
PYTHON_PATH = "/opt/conda/envs/colbert/bin/python"
VOLUME = Volume.from_name("colbert-maintenance", create_if_missing=True)
MOUNT = "/colbert-maintenance"

image = Image.from_dockerfile(f"Dockerfile.compare", gpu=GPU)
app = App("colbert-maintenance-compare")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _data():
    import pandas as pd
    extract_path = f"{MOUNT}/{PROJECT}/data/"
    files = [
        f"{extract_path}/collection.tsv",
        f"{extract_path}/queries.dev.tsv",
        f"{extract_path}/queries.eval.tsv",
        f"{extract_path}/queries.train.tsv",
        f"{extract_path}/triples.train.small.tsv",
    ]

    for f in files[:-1]: 
        df = pd.read_csv(f, sep='\t', header=None)
        print(f)
        print("\t",len(df))
        print("\t",df.head(5))

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _compare_index(path_a, path_b):
    import pandas as pd
    import torch
    import os
    from datasets import load_dataset
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")
    qrels_rows = load_dataset("UKPLab/dapr", f"{dataset_name}-qrels", split="test")

    console = Console(force_terminal=True)

    a_path = f"{MOUNT}/{path_a}/indexing/ConditionalQA/"
    b_path = f"{MOUNT}/{path_b}/indexing/ConditionalQA/"

    console.print(Panel.fit("[bold blue]COMPARING TENSOR DIRECTORIES[/bold blue]"))
    console.print(f"[yellow]Path A:[/yellow] {a_path}")
    console.print(f"[yellow]Path B:[/yellow] {b_path}")

    console.print("\n[bold blue]INDEX FILE NAME COMPARISON[/bold blue]")
    a = os.listdir(a_path)
    b = os.listdir(b_path)

    try:
        for i, f in enumerate(a): assert f == b[i]
        console.print(f"[green]✓ All {len(a)} files match[/green]")
    except:
        console.print("[red]✗ File names don't match[/red]")

    a_pts = [f for f in a if f.endswith(".pt")]
    b_pts = [f for f in b if f.endswith(".pt")]

    #####################################################################################################################
    console.print(f"\n[bold blue]TENSOR SHAPE COMPARISON[/bold blue]")
    #####################################################################################################################

    shape_mismatches = 0

    for i, f in enumerate(a_pts):
        console.print(f"\n[bold]{f}[/bold]")
        a_pt = torch.load(a_path + f)
        b_pt = torch.load(b_path + f)
        
        if isinstance(a_pt, tuple):
            match1 = a_pt[0].shape == b_pt[0].shape
            match2 = a_pt[1].shape == b_pt[1].shape
            console.print(f"  Tensor[0]: [{'green' if match1 else 'red'}]{a_pt[0].shape} vs {b_pt[0].shape}[/{'green' if match1 else 'red'}]")
            console.print(f"  Tensor[1]: [{'green' if match2 else 'red'}]{a_pt[1].shape} vs {b_pt[1].shape}[/{'green' if match2 else 'red'}]")
            if not (match1 and match2):
                shape_mismatches += 1
        else:
            match = a_pt.shape == b_pt.shape
            console.print(f"  Shape: [{'green' if match else 'red'}]{a_pt.shape} vs {b_pt.shape}[/{'green' if match else 'red'}]")
            if not match:
                shape_mismatches += 1

    #####################################################################################################################
    console.print(f"\n[bold blue]TENSOR VALUE COMPARISON[/bold blue]")
    #####################################################################################################################

    value_mismatches = 0

    for i, f in enumerate(a_pts):
        console.print(f"\n[bold]{f}[/bold]")
        a_pt = torch.load(a_path + f)
        b_pt = torch.load(b_path + f)
        
        if isinstance(a_pt, tuple):
            if a_pt[0].shape == b_pt[0].shape:
                match1 = torch.allclose(a_pt[0], b_pt[0])
                console.print(f"  [{'green' if match1 else 'red'}]{'✓' if match1 else '✗'} Tensor[0] values {'match' if match1 else 'differ'}[/{'green' if match1 else 'red'}]")
            else:
                console.print("  [red]✗ Tensor[0] shape mismatch[/red]")
                match1 = False
                
            if a_pt[1].shape == b_pt[1].shape:
                match2 = torch.allclose(a_pt[1], b_pt[1])
                console.print(f"  [{'green' if match2 else 'red'}]{'✓' if match2 else '✗'} Tensor[1] values {'match' if match2 else 'differ'}[/{'green' if match2 else 'red'}]")
            else:
                console.print("  [red]✗ Tensor[1] shape mismatch[/red]")
                match2 = False
                
            if not (match1 and match2):
                value_mismatches += 1
        else:
            if a_pt.shape == b_pt.shape:
                match = torch.allclose(a_pt, b_pt)
                console.print(f"  [{'green' if match else 'red'}]{'✓' if match else '✗'} Values {'match' if match else 'differ'}[/{'green' if match else 'red'}]")
            else:
                console.print("  [red]✗ Shape mismatch[/red]")
                match = False
                
            if not match:
                value_mismatches += 1

    console.print(Panel.fit(
        f"[bold]Files processed:[/bold] {len(a_pts)}\n"
        f"[bold]Shape matches:[/bold] [{'green' if shape_mismatches == 0 else 'red'}]{len(a_pts) - shape_mismatches}/{len(a_pts)}[/{'green' if shape_mismatches == 0 else 'red'}]\n"
        f"[bold]Value matches:[/bold] [{'green' if value_mismatches == 0 else 'red'}]{len(a_pts) - value_mismatches}/{len(a_pts)}[/{'green' if value_mismatches == 0 else 'red'}]\n\n"
        f"[{'green' if shape_mismatches == 0 and value_mismatches == 0 else 'red'}]{'🎉 ALL TENSORS ARE IDENTICAL! 🎉' if shape_mismatches == 0 and value_mismatches == 0 else '⚠️  DIFFERENCES FOUND ⚠️'}[/{'green' if shape_mismatches == 0 and value_mismatches == 0 else 'red'}]",
        title="[bold blue]FINAL SUMMARY[/bold blue]"
    ))

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _compare_search(path_a, path_b):
    from ranx import Qrels, Run, evaluate
    from datasets import load_dataset
    import pandas as pd
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console(force_terminal=True)

    dataset_name = "ConditionalQA"
    passages = load_dataset("UKPLab/dapr", f"{dataset_name}-corpus", split="test")
    queries = load_dataset("UKPLab/dapr", f"{dataset_name}-queries", split="test")
    qrels_rows = load_dataset("UKPLab/dapr", f"{dataset_name}-qrels", split="test")

    a_path = f"{MOUNT}/{path_a}/search/ConditionalQA.tsv"
    b_path = f"{MOUNT}/{path_b}/search/ConditionalQA.tsv"

    console.print(Panel.fit("[bold blue]COMPARING RETRIEVAL RESULTS[/bold blue]"))
    console.print(f"[yellow]Path A:[/yellow] {a_path}")
    console.print(f"[yellow]Path B:[/yellow] {b_path}")

    #####################################################################################################################
    console.print("\n[bold blue]METRICS COMPARISON[/bold blue]")
    #####################################################################################################################

    def _metrics(fpath):
        ranking = pd.read_csv(fpath, sep='\t', header=None, names=['qid', 'pid', 'rank', 'score'])
        ranking['pid'] = ranking['pid'].apply(lambda x: passages[x]['_id'])
        results = {qid: dict(zip(group['pid'], group['score'])) for qid, group in ranking.groupby('qid')}

        qrels = {}
        for qrel_row in qrels_rows:
            qid = qrel_row["query_id"]
            pid = qrel_row["corpus_id"]
            rel = qrel_row["score"]
            qrels.setdefault(qid, {})
            qrels[qid][pid] = rel

        qrels = Qrels(qrels)
        run = Run(results)
        metrics = evaluate(qrels, run, ["recall@10", "mrr@10"])

        return metrics['recall@10'], metrics['mrr@10'], dict(run.scores)

    a_mr, a_mrr, a_m = _metrics(a_path)
    b_mr, b_mrr, b_m = _metrics(b_path)
    
    try:
        assert a_mr == b_mr
        console.print(f"[green]✓ Mean Recall@10 Matches[/green]")
    except:
        console.print("[red]✗ Mean Recall@10 Doesn't Match[/red]")
        console.print(f"\tA: [red]{a_mr}[/red]")
        console.print(f"\tB: [red]{b_mr}[/red]")


    try:
        assert a_mrr == b_mrr
        console.print(f"[green]✓ Mean MRR@10 Matches[/green]")
    except:
        console.print("[red]✗ Mean MRR@10 Doesn't Match[/red]")
        console.print(f"\tA: [red]{a_mrr}[/red]")
        console.print(f"\tB: [red]{b_mrr}[/red]")

    #####################################################################################################################
    console.print("\n[bold red]AVG NUMBER OF DIFFERENT PASSAGES RETRIEVED[/bold red]")
    #####################################################################################################################

    a_ranking = pd.read_csv(a_path, sep='\t', header=None, names=['qid', 'pid', 'rank', 'score'])
    b_ranking = pd.read_csv(b_path, sep='\t', header=None, names=['qid', 'pid', 'rank', 'score'])

    count = 0
    
    for qid in set(a_ranking['qid'].unique()) | set(b_ranking['qid'].unique()):
        a_pids = set(a_ranking[a_ranking['qid'] == qid]['pid'])
        b_pids = set(b_ranking[b_ranking['qid'] == qid]['pid'])
        count += len(a_pids.symmetric_difference(b_pids))


    console.print(f"[yellow]{count / len(queries)}[/yellow]")

    #####################################################################################################################
    console.print("\n[bold blue]QUERY-LEVEL RECALL@10 DIFFERENCE[/bold blue]")
    #####################################################################################################################

    diffs = []
    metric = "recall@10"
    for qid in a_m[metric].keys():
        if b_m[metric][qid] < a_m[metric][qid]: diffs.append('Increase')
        if b_m[metric][qid] == a_m[metric][qid]: diffs.append('Equal')
        if b_m[metric][qid] < a_m[metric][qid]: diffs.append('Decrease') 

    diffs = pd.Series(diffs)
    console.print(diffs.value_counts())


    #####################################################################################################################
    console.print("\n[bold blue]QUERY-LEVEL MRR@10 DIFFERENCE[/bold blue]")
    #####################################################################################################################

    diffs = []
    metric = "mrr@10"
    for qid in a_m[metric].keys():
        if b_m[metric][qid] < a_m[metric][qid]: diffs.append('Increase')
        if b_m[metric][qid] == a_m[metric][qid]: diffs.append('Equal')
        if b_m[metric][qid] < a_m[metric][qid]: diffs.append('Decrease') 

    diffs = pd.Series(diffs)
    console.print(diffs.value_counts())

@app.local_entrypoint()
def main():
    #_compare_index.remote(path_a=PATH_A, path_b=PATH_B)
    #_compare_search.remote(path_a=PATH_A, path_b=PATH_B)
    #_data.remote()