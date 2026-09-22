"""Make the existing GPTCache ONNX model accept dynamic batch sizes.

Instead of re-exporting from PyTorch (which requires compatible torch +
transformers + optimum versions), this script:

  1. Downloads the original static-batch model.onnx from HuggingFace Hub
  2. Uses the `onnx` package to rewrite the input/output shape annotations
     so the batch dimension is dynamic ("batch_size") instead of hardcoded 1
  3. Saves the result to --out-dir/model.onnx

The underlying ONNX ops are already batch-agnostic (matmul, attention, etc.)
in ALBERT/BERT models — only the shape *metadata* needs changing.

Requirements: onnx (already installed), huggingface_hub (already installed)

Usage
-----
  python scripts/export_onnx_dynamic.py [--out-dir ./onnx_dynamic]
"""

import argparse
import os
import sys


def make_dynamic(model):
    """Replace static batch dimension (value=1) with a dynamic symbol."""
    import onnx

    changed = 0
    for tensor in list(model.graph.input) + list(model.graph.output):
        shape = tensor.type.tensor_type.shape
        if shape and len(shape.dim) > 0:
            d = shape.dim[0]
            if d.dim_value == 1:          # static batch size 1 → make dynamic
                d.ClearField("dim_value")
                d.dim_param = "batch_size"
                changed += 1
            elif d.dim_value == 0 and not d.dim_param:
                d.dim_param = "batch_size"  # already unknown, just name it
                changed += 1
    return model, changed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="onnx_dynamic",
                   help="Directory to write the patched model.onnx")
    p.add_argument("--hub-model", default="GPTCache/paraphrase-albert-onnx",
                   help="HuggingFace repo containing the original model.onnx")
    args = p.parse_args()

    try:
        import onnx
    except ImportError:
        print("ERROR: onnx not installed.  Run:  pip install onnx")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.  Run:  pip install huggingface-hub")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "model.onnx")

    print(f"Downloading original model.onnx from '{args.hub_model}' ...")
    src_path = hf_hub_download(repo_id=args.hub_model, filename="model.onnx")
    print(f"  Cached at: {src_path}")

    print("Loading ONNX model ...")
    model = onnx.load(src_path)

    print("Patching batch dimension to dynamic ...")
    model, n_changed = make_dynamic(model)
    print(f"  Changed {n_changed} tensor shape(s)")

    onnx.checker.check_model(model)
    onnx.save(model, out_path)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDone. Patched model saved to: {out_path} ({size_mb:.1f} MB)")

    # Quick sanity check with onnxruntime
    print("\nRunning sanity check (batch=4) ...")
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path)
        dummy = np.zeros((4, 512), dtype=np.int64)
        out = sess.run(None, {
            "input_ids": dummy,
            "attention_mask": dummy,
            "token_type_ids": dummy,
        })
        print(f"  Output shape: {out[0].shape}  (expected (4, 512, 768))")
        print("  Sanity check PASSED ✓")
    except Exception as e:
        print(f"  Sanity check failed: {e}")
        print("  The model was saved but may not accept dynamic batches.")

    abs_dir = os.path.abspath(args.out_dir)
    print()
    print("Next steps (PowerShell):")
    print(f'  $env:GPTCACHE_ONNX_MODEL_DIR = "{abs_dir}"')
    print("  python examples\\benchmark\\benchmark_qqp.py --scale 10000 "
          "--encoder auto --data qqp --repeats 3 --threads 1 --workdir bench_real_10k")


if __name__ == "__main__":
    main()
