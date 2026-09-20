"""Build-time export of the embedding model to ONNX.

Runs only inside the Docker builder stage (needs optimum + torch, which never
reach the runtime image). Exports antoinelouis/french-me5-small to a single
fp32 ONNX graph and copies the fast tokenizer alongside it.

Usage: python export_onnx.py <output_dir>
"""

import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from optimum.onnxruntime import ORTModelForFeatureExtraction

MODEL_ID = "antoinelouis/french-me5-small"
# Pin the model revision for reproducible builds (and to satisfy B615).
MODEL_REVISION = "1605fe3b64cb22dde93f30e8fbfb1889742d2753"


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = ORTModelForFeatureExtraction.from_pretrained(
        MODEL_ID, export=True, revision=MODEL_REVISION
    )
    model.save_pretrained(out)

    tokenizer_path = hf_hub_download(MODEL_ID, "tokenizer.json", revision=MODEL_REVISION)
    shutil.copy(tokenizer_path, out / "tokenizer.json")

    print(f"Exported ONNX model + tokenizer to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "onnx_model")
