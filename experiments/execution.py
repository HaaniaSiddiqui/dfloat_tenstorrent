"""
Mock grouped execution schedule.

Loads exported grouped compressed format and simulates:
- load one group
- decode exponent stream
- reconstruct grouped BF16 weights
- split back into original layer tensors

This is a scheduling / format validation step.
"""

import os
import sys
import json
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.utils import (
    reconstruct_bf16,
    decode_exponents,
    load_model,
)

EXPORT_DIR = "artifacts/grouped_export"


def rebuild_codec_from_freq(freq_dict: dict):
    from dahuffman import HuffmanCodec
    freq_int = {int(k): int(v) for k, v in freq_dict.items()}
    return HuffmanCodec.from_frequencies(freq_int)


def load_original_linear_weights(model_name: str):
    from core.utils import get_linear_layers
    model = load_model(model_name)
    linear_layers = get_linear_layers(model)
    return {name: weight for name, weight in linear_layers}


def execute_one_group(group_dir: Path, original_weights: dict, max_layers_to_test: int = 3):
    with open(group_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    sign_mantissa = torch.load(group_dir / "sign_mantissa.pt", map_location="cpu")

    with open(group_dir / "encoded_exponents.bin", "rb") as f:
        encoded = f.read()

    codec = rebuild_codec_from_freq(metadata["exponent_frequencies"])

    total_vals = metadata["total_vals"]
    decoded_exponent = codec.decode(encoded)
    decoded_exponent = torch.tensor(decoded_exponent[:total_vals], dtype=torch.uint8)

    _, reconstructed_weight = reconstruct_bf16(decoded_exponent, sign_mantissa)

    split_positions = metadata["split_positions"]
    layers = metadata["layers"]

    chunks = []
    start = 0
    for layer in layers:
        end = start + layer["num_vals"]
        chunk = reconstructed_weight[start:end].reshape(layer["shape"])
        chunks.append((layer["name"], chunk))
        start = end

    print("=" * 72)
    print(f"GROUP EXECUTION: {metadata['group_name']}")
    print(f"  total_vals              = {metadata['total_vals']}")
    print(f"  num_layers              = {metadata['num_layers']}")
    print(f"  avg_exp_bits            = {metadata['avg_exp_bits']:.4f}")

    for name, rec_w in chunks[:max_layers_to_test]:
        orig_w = original_weights[name]

        out_features, in_features = orig_w.shape
        x = torch.randn(4, in_features, dtype=torch.bfloat16)

        y_orig = x @ orig_w.t()
        y_rec = x @ rec_w.t()

        exact = torch.equal(y_orig, y_rec)
        close = torch.allclose(y_orig.float(), y_rec.float(), atol=0, rtol=0)

        print(f"  {name}")
        print(f"    matmul_exact_match    = {exact}")
        print(f"    matmul_close_match    = {close}")

        if not exact:
            raise RuntimeError(f"Execution mismatch in group {metadata['group_name']} for layer {name}")


def main():
    export_root = Path(EXPORT_DIR)

    if not export_root.exists():
        raise FileNotFoundError(f"Export directory not found: {export_root}")

    with open(export_root / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    model_name = manifest["model_name"]
    original_weights = load_original_linear_weights(model_name)

    for group in manifest["groups"]:
        group_dir = Path(group["group_dir"])
        execute_one_group(group_dir, original_weights)

    print("\n" + "=" * 72)
    print("Mock group execution completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()

