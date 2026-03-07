"""
Run the full pipeline in order: data load -> train baselines -> train BERT -> eval LLM ->
evaluate -> build networks -> analyze networks.
BERT training uses all available GPUs automatically (accelerate launch --num_processes=N).
"""

import os
import sys
import subprocess
import warnings

# Reduce noisy output: transformers load reports and key mismatches
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Suppress PyTorch pin_memory and NetworkX divide-by-zero warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")


def _num_gpus():
    """Number of GPUs to use (1 if none or CPU-only)."""
    try:
        import torch
        n = torch.cuda.device_count()
        return max(1, n)
    except Exception:
        return 1

STEPS = [
    ("data_load.py", "Step 1/8: Load data and label alignment"),
    ("train_baselines.py", "Step 2/8: Train TF-IDF+LR and TF-IDF+SGD"),
    ("train_bert.py", "Step 3/8: Fine-tune BERT on Reddit"),
    ("eval_llm.py", "Step 4/8: LLM/keyword baseline"),
    ("evaluate.py", "Step 5/8: Evaluate all models"),
    ("build_networks.py", "Step 6/8: Build emotion networks"),
    ("analyze_networks.py", "Step 7/8: Network analysis"),
    ("save_figures.py", "Step 8/8: Save figures (confusion matrix + networks)"),
]


def run(script, description):
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    r = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        print("FAILED:", script)
        sys.exit(r.returncode)


def run_accelerate(script, description):
    """Run script with 'accelerate launch', using all available GPUs (no accelerate config needed)."""
    n = _num_gpus()
    print("\n" + "=" * 60)
    print(description)
    if n > 1:
        print("Using {} GPUs (accelerate launch --num_processes={})".format(n, n))
    print("=" * 60)
    cmd = [
        "accelerate", "launch",
        "--num_processes", str(n),
        "--num_machines", "1",
        "--mixed_precision", "no",
        script,
    ]
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if r.returncode != 0:
        print("FAILED:", script)
        sys.exit(r.returncode)


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    for script, desc in STEPS:
        if script == "train_bert.py":
            run_accelerate(script, desc)
        else:
            run(script, desc)
    print("\nDone. Outputs: outputs/results/ and outputs/figures/")
