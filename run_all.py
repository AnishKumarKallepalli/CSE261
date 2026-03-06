"""
Run the full pipeline in order: data load -> train baselines -> train BERT -> eval LLM ->
evaluate -> build networks -> analyze networks.
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


if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    for script, desc in STEPS:
        run(script, desc)
    print("\nDone. Outputs: outputs/results/ and outputs/figures/")
