import modal
from modal import Image, App, Secret, Volume
import datetime
import os

PATH_A = os.environ.get("PATH_A", "")
PATH_B = os.environ.get("PATH_B", "")
DEFAULT = os.environ.get("DEFAULT", "False")
DEFAULT = DEFAULT == "True"
GPU = os.environ.get("GPU", "L4") 
PYTHON_PATH = "/opt/conda/envs/colbert/bin/python"
VOLUME = Volume.from_name("colbert-maintenance", create_if_missing=True)
MOUNT = "/colbert-maintenance"
SWAP = os.environ.get("SWAP", False)
SWAP = SWAP == "True"
image = Image.from_dockerfile(f"Dockerfile.compare", gpu=GPU)
app = App("colbert-maintenance-compare")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _subtract(root_a, root_b, default):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{MOUNT}/{root_a}/{fn}")
        b = torch.load(f"{MOUNT}/{root_b}/{fn}")
        console.print(_print(f"{fn} torch.allclose:", _close(a, b), True))
        console.print(_print(f"{fn} torch.equal:", torch.equal(a,b), True))

    _compare("subtract_r.pt")


@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _residuals(root_a, root_b, default):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)
    
    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{MOUNT}/{root_a}/{fn}")
        b = torch.load(f"{MOUNT}/{root_b}/{fn}")
        console.print(_print(f"{fn} torch.allclose:", _close(a, b), True))
        console.print(_print(f"{fn} torch.equal:", torch.equal(a, b), True))

    for o in ["residuals_", "codes_", "batch", "centroids_"]: 
        for i in [0, 1, 2]: _compare(f"{o}_{i}.pt")

    _compare("lookup_centroids_self_centroids.pt")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _sort(root_a, root_b):
    import torch
    from rich.console import Console
    console = Console(force_terminal=True)

    a = torch.load(f"{MOUNT}/{root_a}/randint_t.pt", weights_only=False)
    b = torch.load(f"{MOUNT}/{root_b}/randint_t.pt", weights_only=False)
    console.print(f"randint_t: {torch.equal(a,b)}")
    console.print(a)
    console.print(b)

    a = torch.load(f"{MOUNT}/{root_a}/randint_t_sorted.pt", weights_only=False)
    b = torch.load(f"{MOUNT}/{root_b}/randint_t_sorted.pt", weights_only=False)
    console.print(f"randint_t sorted indices: {torch.equal(a.indices,b.indices)}")
    console.print(f"randint_t sorted values: {torch.equal(a.values,b.values)}")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _lse(root_a, root_b, default):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    # a = torch.load(f"{MOUNT}/{root_a}/local_sample_embs.pt")
    # b = torch.load(f"{MOUNT}/{root_b}/local_sample_embs.pt")
    # # 53768
    # for idx in range(53700, 53800): console.print(idx, _close(a[:idx], b[:idx]))

    for idx in range(29):
        a = torch.load(f"{MOUNT}/{root_a}/lse_outputs_dict_{idx}.pt")
        b = torch.load(f"{MOUNT}/{root_b}/lse_outputs_dict_{idx}.pt")

        print(f"dict {idx}")
        for i in range(len(a.keys())):
            a_ = a[f"{i}"]
            b_ = b[f"{i}"]
            #console.print(f"\tLayer {i}", _close(a_, b_))
            assert _close(a_, b_)


@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _norm(root_a, root_b):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{MOUNT}/{root_a}/{fn}")
        b = torch.load(f"{MOUNT}/{root_b}/{fn}")
        console.print(_print(f"{fn} torch.allclose:", _close(a, b), True))
        console.print(_print(f"{fn} torch.equal:", torch.equal(a, b), True))
        # console.print(_print(f"{fn} MAD: {torch.abs(a - b).float().mean()}", torch.abs(a - b).float().mean() == 0, True))

    _compare("t.pt")
    _compare("half_t.pt")
    _compare("norm.pt")
    _compare("half_norm.pt")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _centroids(root_a, root_b):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{MOUNT}/{root_a}/{fn}")
        b = torch.load(f"{MOUNT}/{root_b}/{fn}")
        # print(a.min(), a.max())
        console.print(_print(f"{fn} torch.allclose:", _close(a, b), True))
        # console.print(_print(f"{fn} MAD: {torch.abs(a - b).float().mean()}", torch.abs(a - b).float().mean() == 0, True))
        console.print(_print(f"{fn} torch.equal:", torch.equal(a, b), True))


    _compare("prenorm_centroids.pt")
    _compare("postnorm_centroids.pt")
    _compare("half_centroids.pt")
    _compare("centroids.pt")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _remove_dense(root_a, root_b):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    a = torch.load(f"{root_a}/rd_outputs_dict.pt")
    b = torch.load(f"{root_b}/rd_outputs_dict.pt")

    for i in range(10):
        a_ = a[f"{i}"]
        b_ = b[f"{i}"]
        print(f"Layer {i}", torch.abs(a_ - b_).float().mean())

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{root_a}/{fn}")
        b = torch.load(f"{root_b}/{fn}")
        console.print(_print(f"{fn} torch.allclose:", torch.allclose(a, b), True))

    _compare("D_rd_bert.pt")
    _compare("D_rd_linear.pt")
    _compare("D_rd_mask.pt")
    _compare("D_rd_norm.pt")

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _amp_bert(root_a, root_b):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    a = torch.load(f"{MOUNT}/{root_a}/amp_outputs_dict.pt")
    b = torch.load(f"{MOUNT}/{root_b}/amp_outputs_dict.pt")

    for i in range(len(a.keys())):
        a_ = a[f"{i}"]
        b_ = b[f"{i}"]
        console.print(f"Layer {i}", _close(a_, b_))

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{MOUNT}/{root_a}/{fn}")
        b = torch.load(f"{MOUNT}/{root_b}/{fn}")
        console.print(_print(f"{fn} torch.allclose:", _close(a, b), True))

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _batches(root_a, root_b):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _load(fn):
        print("\n")
        console.print(f"[bold white]{fn}[/bold white]")
        a = torch.load(f"{root_a}/{fn}", weights_only=False)
        b = torch.load(f"{root_b}/{fn}", weights_only=False)
        return a, b

    a,b = _load("batches.pt")

    # for i in range(len(a)): 
    #     flag = (a[i][0] == b[i][0]).float().mean() == 1
    #     console.print(_print(f"batch {i}: {(a[i][0] == b[i][0]).float().mean()}", flag))
    idx = -1
    for i in range(8): console.print((a[idx][0][i] == b[idx][0][i]).float().mean())
        # console.print(torch.abs(a[idx][0][i] - b[idx][0][i]).float().mean())

    # console.print("[bold white]Final batch[/bold white]")
    # console.print("[white]root_a[/white]")
    # console.print(a[-1][0].shape)

    # console.print("[white]root_b[/white]")
    # console.print(b[-1][0].shape)

@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _bert(root_a, root_b, default):
    import torch
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    a = torch.load(f"{MOUNT}/{root_a}/bert_weights.pt")
    b = torch.load(f"{MOUNT}/{root_b}/bert_weights.pt")
    for key in a.keys(): assert _close(a[key], b[key])

    a = torch.load(f"{MOUNT}/{root_a}/outputs_dict.pt")
    b = torch.load(f"{MOUNT}/{root_b}/outputs_dict.pt")

    for i in range(len(a.keys())):
        a_ = a[f"{i}"]
        b_ = b[f"{i}"]
        console.print(f"Layer {i}", _close(a_, b_))

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"
    def _compare(fn):
        a = torch.load(f"{MOUNT}/{root_a}/{fn}")
        b = torch.load(f"{MOUNT}/{root_b}/{fn}")
        console.print(_print(f"{fn} torch.allclose:", _close(a, b), True))

    _compare("D_bert.pt")
    _compare("D_linear.pt")
    _compare("D_mask.pt")
    _compare("D_norm.pt")


@app.function(gpu=GPU, image=image, timeout=3600,
              volumes={MOUNT: VOLUME},
              max_containers=1)
def _intermediate(root_a, root_b, default=False):
    import pandas as pd
    import torch
    from torch import tensor
    import os
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype
        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        print(gtype, atol, rtol)
        return torch.allclose(a, b, rtol=rtol, atol=atol)

    def _load(fn):
        print("\n")
        console.print(f"[bold white]{fn}[/bold white]")
        a = torch.load(f"{MOUNT}/{root_a}/{fn}", weights_only=False)
        b = torch.load(f"{MOUNT}/{root_b}/{fn}", weights_only=False)
        return a, b

    def _print(string, flag, print_flag=False): return f"[{'green' if flag else 'red'}]{string}\t{flag if print_flag else ''}[{'/green' if flag else '/red'}]"

    a,b = _load("sample_pids.pt")
    for item in a: 
        print(type(item))
        break
    flag = _close(tensor(list(a), dtype=torch.int32),tensor(list(b), dtype=torch.int32))
    console.print(_print("torch.allclose:", flag, True))

    a,b = _load("num_passages.pt")
    flag = a == b
    console.print(f"[{'green' if flag else 'red'}]a==b[{'/green' if flag else '/red'}]: {flag}")

    def _compare(fn):
        a,b = _load(fn)
        # flag = torch.allclose(a,b)
        flag = _close(a, b, default=default)
        console.print(_print("torch.allclose:", flag, True))
        console.print(_print(f"Mean Acc:\t{(a == b).float().mean()}", flag))
        console.print(_print(f"MAD:\t\t{torch.abs(a-b).float().mean()}", flag))
        console.print(_print(f"Max Abs Diff:\t{torch.abs(a-b).float().max()}", flag))

    _compare("local_sample_embs.pt")
    _compare("centroids.pt")
    _compare("bucket_cutoffs.pt")
    _compare("bucket_weights.pt")
    _compare("avg_residual.pt")
    _compare("sample.pt")
    _compare("sample_heldout.pt")
    # _compare("embs.pt")

    a,b = _load("doclens.pt")
    flag = a == b
    console.print(f"[{'green' if flag else 'red'}]a==b[{'/green' if flag else '/red'}]: {flag}")

    a,b = _load("codes.pt")
    # flag = torch.allclose(a.values, b.values)
    flag = _close(a.values, b.values, default=default)
    console.print(_print(f"torch.allclose values:", flag, True))
    # flag = torch.allclose(a.indices, b.indices)
    flag = _close(a.indices, b.indices, default=default)
    console.print(_print(f"torch.allclose indices:", flag, True))

    _compare("ivf.pt")
    _compare("values.pt")

    # a,b = _load("tensorize_output.pt")
    # print("token ids")
    # for i in range(len(a[0])): 
    #     flag = (a[0][i][0] == b[0][i][0]).float().mean() == 1
    #     console.print(_print(i, flag, True))
    # print("masks")
    # for i in range(len(a[0])): 
    #     flag = (a[0][i][1] == b[0][i][1]).float().mean() == 1
    #     console.print(_print(i, flag, True))

    a,b = _load("batches.pt")
    for i in range(len(a)): 
        flag = _close(a[i][0], b[i][0], default=default)
        # console.print(a[i][0].shape)
        console.print(i, _print(f"torch.allclose values:", flag, True))
        # flag = (a[i][0] == b[i][0]).float().mean() == 1
        # console.print(_print(f"batch {i}: {(a[i][0] == b[i][0]).float().mean()}", flag))

    _compare("D.pt")
    for f in ["embs_0.pt", "embs_1.pt", "embs_2.pt"]: _compare(f)

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
def _index(path_a, path_b, default):
    print(f"Default: {default}")
    import pandas as pd
    import torch
    import os
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    console = Console(force_terminal=True)

    def _close(a, b, default=False):
        gtype = a.dtype

        if gtype in [torch.uint8, torch.int32, torch.int64]:
            if a.shape == b.shape: return torch.equal(a,b)
            return False

        if not default:
            if gtype == torch.float32:
                atol, rtol = 1e-6, 1e-5
            elif gtype == torch.bfloat16:
                atol, rtol = 1e-3, 1e-2
            else:
                atol, rtol = 1e-4, 1e-3
        else:
            atol, rtol = 1e-8, 1e-5
        return torch.allclose(a, b, rtol=rtol, atol=atol)

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
                # match1 = torch.allclose(a_pt[0], b_pt[0])
                match1 = _close(a_pt[0], b_pt[0], default=default)
                console.print(f"  [{'green' if match1 else 'red'}]{'✓' if match1 else '✗'} Tensor[0] values {'match' if match1 else 'differ'}[/{'green' if match1 else 'red'}]")
            else:
                console.print(a_pt[0].dtype)
                console.print("  [red]✗ Tensor[0] shape mismatch[/red]")
                match1 = False
                
            if a_pt[1].shape == b_pt[1].shape:
                # match2 = torch.allclose(a_pt[1], b_pt[1])
                match2 = _close(a_pt[1], b_pt[1], default=default)
                console.print(f"  [{'green' if match2 else 'red'}]{'✓' if match2 else '✗'} Tensor[1] values {'match' if match2 else 'differ'}[/{'green' if match2 else 'red'}]")
            else:
                console.print(a_pt[1].dtype)
                console.print("  [red]✗ Tensor[1] shape mismatch[/red]")
                match2 = False
                
            if not (match1 and match2):
                value_mismatches += 1
        else:
            if a_pt.shape == b_pt.shape:
                # match = torch.allclose(a_pt, b_pt)
                match = _close(a_pt, b_pt, default=default)
                console.print(f"  [{'green' if match else 'red'}]{'✓' if match else '✗'} Values {'match' if match else 'differ'}[/{'green' if match else 'red'}]")
            else:
                console.print(a_pt.dtype)
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
def _search(path_a, path_b, swap=False):
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

    if swap==True: b_path = f"{MOUNT}/{path_b}/search/ConditionalQA_swap.tsv"

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
        console.print(f"\tA: [green]{a_mr}[/green]")
        console.print(f"\tB: [green]{b_mr}[/green]")
    except:
        console.print("[red]✗ Mean Recall@10 Doesn't Match[/red]")
        console.print(f"\tA: [red]{a_mr}[/red]")
        console.print(f"\tB: [red]{b_mr}[/red]")
        console.print(b_mr - a_mr)


    try:
        assert a_mrr == b_mrr
        console.print(f"[green]✓ Mean MRR@10 Matches[/green]")
        console.print(f"\tA: [green]{a_mrr}[/green]")
        console.print(f"\tB: [green]{b_mrr}[/green]")
    except:
        console.print("[red]✗ Mean MRR@10 Doesn't Match[/red]")
        console.print(f"\tA: [red]{a_mrr}[/red]")
        console.print(f"\tB: [red]{b_mrr}[/red]")
        console.print(b_mrr - a_mrr)

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
    print(len(a_m[metric].keys()))
    for qid in a_m[metric].keys():
        if b_m[metric][qid] > a_m[metric][qid]: diffs.append('Increase')
        if b_m[metric][qid] == a_m[metric][qid]: diffs.append('Equal')
        if b_m[metric][qid] < a_m[metric][qid]: diffs.append('Decrease') 

    diffs = pd.Series(diffs)
    console.print(diffs.value_counts())


    #####################################################################################################################
    console.print("\n[bold blue]QUERY-LEVEL MRR@10 DIFFERENCE[/bold blue]")
    #####################################################################################################################

    diffs = []
    metric = "mrr@10"
    print(len(a_m[metric].keys()))
    for qid in a_m[metric].keys():
        if b_m[metric][qid] > a_m[metric][qid]: diffs.append('Increase')
        if b_m[metric][qid] == a_m[metric][qid]: diffs.append('Equal')
        if b_m[metric][qid] < a_m[metric][qid]: diffs.append('Decrease') 

    diffs = pd.Series(diffs)
    console.print(diffs.value_counts())

@app.local_entrypoint()
def main():
    # _index.remote(path_a=PATH_A, path_b=PATH_B, default=DEFAULT)
    _search.remote(path_a=PATH_A, path_b=PATH_B, swap=SWAP)
    # _data.remote()
    # _intermediate.remote(root_a=PATH_A, root_b=PATH_B, default=DEFAULT)
    # _bert.remote(root_a=PATH_A, root_b=PATH_B, default=DEFAULT)
    # _batches.remote(root_a=PATH_A, root_b=PATH_B)
    # _amp_bert.remote(root_a=PATH_A, root_b=PATH_B)
    # _remove_dense.remote(root_a=PATH_A, root_b=PATH_B)
    # _centroids.remote(root_a=PATH_A, root_b=PATH_B)
    # _norm.remote(root_a=PATH_A, root_b=PATH_B)
    # _lse.remote(root_a=PATH_A, root_b=PATH_B, default=DEFAULT)
    # _sort.remote(root_a=PATH_A, root_b=PATH_B)
    # _residuals.remote(root_a=PATH_A, root_b=PATH_B, default=DEFAULT)
    # _subtract.remote(root_a=PATH_A, root_b=PATH_B, default=DEFAULT)