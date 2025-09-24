import subprocess
import time
import modal
from modal import Image, App, Secret, Volume
import datetime
import os

SOURCE = os.environ.get("SOURCE", "")
VOLUME = Volume.from_name("colbert-maintenance", create_if_missing=True)
MOUNT = "/colbert-maintenance"
image = Image.from_dockerfile(f"Dockerfile.{SOURCE}", gpu="L4")

app = App("jupyter-tunnel", image=image.pip_install("jupyter"))
JUPYTER_TOKEN = "97201" # some list of characters you'll enter when accessing the Modal tunnel

@app.function(max_containers=1, volumes={MOUNT: VOLUME}, timeout=30_000, gpu="L4")
def run_jupyter_with_fasthtml(timeout: int):
    jupyter_port = 8888
    fasthtml_port = 8000
    
    # Nested with statements
    with modal.forward(jupyter_port) as jupyter_tunnel:
        with modal.forward(fasthtml_port) as fasthtml_tunnel:
            
            # Start Jupyter
            jupyter_process = subprocess.Popen(
                [
                    "jupyter",
                    "notebook",
                    "--no-browser",
                    "--allow-root",
                    "--ip=0.0.0.0",
                    f"--port={jupyter_port}",
                    "--NotebookApp.allow_origin='*'",
                    "--NotebookApp.allow_remote_access=1",
                ],
                env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
            )

            print(f"Jupyter available at => {jupyter_tunnel.url}")
            print(f"FastHTML apps will be available at => {fasthtml_tunnel.url}")
            print(f"Use port {fasthtml_port} in your FastHTML apps")

            try:
                end_time = time.time() + timeout
                while time.time() < end_time:
                    time.sleep(5)
                print(f"Reached end of {timeout} second timeout period. Exiting...")
            except KeyboardInterrupt:
                print("Exiting...")
            finally:
                jupyter_process.kill()

# @app.function(max_containers=1, volumes={MOUNT: VOLUME}, timeout=30_000, gpu="L4")
# def run_jupyter(timeout: int):
#     jupyter_port = 8888
#     with modal.forward(jupyter_port) as tunnel, modal.forward(8000) as fasthtml_tunnel::
#         jupyter_process = subprocess.Popen(
#             [
#                 "jupyter",
#                 "notebook",
#                 "--no-browser",
#                 "--allow-root",
#                 "--ip=0.0.0.0",
#                 f"--port={jupyter_port}",
#                 "--NotebookApp.allow_origin='*'",
#                 "--NotebookApp.allow_remote_access=1",
#             ],
#             env={**os.environ, "JUPYTER_TOKEN": JUPYTER_TOKEN},
#         )

#         print(f"Jupyter available at => {tunnel.url}")

#         try:
#             end_time = time.time() + timeout
#             while time.time() < end_time:
#                 time.sleep(5)
#             print(f"Reached end of {timeout} second timeout period. Exiting...")
#         except KeyboardInterrupt:
#             print("Exiting...")
#         finally:
#             jupyter_process.kill()


@app.local_entrypoint()
def main(timeout: int = 10_000):
    # run_jupyter.remote(timeout=timeout)
    run_jupyter_with_fasthtml.remote(timeout=timeout)