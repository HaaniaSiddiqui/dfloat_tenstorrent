"""
Export grouped compressed format for shard-aware, family-aware DFloat-style storage.

Grouping policy:
- shard: fake hardware partition
- family: attn / mlp / lm_head

For each group, we save:
- sign_mantissa.pt
- encoded_exponents.bin
- metadata.json

This is the next step toward a hardware-facing format.
"""

import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.utils import (
    split_bf16_bits,
    build_codec,
    avg_bits,
    encode_exponents,
    load_model,
    get_linear_layers,
    get_layer_type,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "artifacts/grouped_export"


def assign_shard(layer_name: str) -> str:
    if layer_name == "lm_head":
        return "shard_3"

    if layer_name.startswith("model.layers."):
        parts = layer_name.split(".")
        layer_idx = int(parts[2])

        if 0 <= layer_idx <= 5:
            return "shard_0"
        elif 6 <= layer_idx <= 11:
            return "shard_1"
        elif 12 <= layer_idx <= 17:
            return "shard_2"
        else:
            return "shard_3"

    return "shard_other"


def get_family(layer_type: str) -> str:
    if layer_type in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attn"
    if layer_type in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    if layer_type == "lm_head":
        return "lm_head"
    return "other"


def build_grouped_layers(model_name=MODEL_NAME):
    model = load_model(model_name)
    linear_layers = get_linear_layers(model)

    grouped = defaultdict(list)

    for name, weight in linear_layers:
        layer_type = get_layer_type(name)
        family = get_family(layer_type)
        shard = assign_shard(name)
        key = (shard, family)

        grouped[key].append({
            "name": name,
            "type": layer_type,
            "family": family,
            "shard": shard,
            "shape": list(weight.shape),
            "num_vals": int(weight.numel()),
            "weight": weight,
        })

    return grouped


def export_one_group(group_dir: Path, shard: str, family: str, group_layers: list[dict]):
    flat_weights = [x["weight"].reshape(-1) for x in group_layers]
    combined_weight = torch.cat(flat_weights, dim=0)

    _, exponent, sign_mantissa = split_bf16_bits(combined_weight)

    codec, freq_dict = build_codec(exponent)
    avg_exp_bits = avg_bits(codec, freq_dict)

    encoded = encode_exponents(exponent, codec)

    split_positions = []
    running = 0
    for item in group_layers[:-1]:
        running += item["num_vals"]
        split_positions.append(running)

    sign_mantissa_bytes = int(sign_mantissa.numel())
    encoded_exp_bytes = int(len(encoded))
    split_metadata_bytes = int(len(split_positions) * 8)

    total_vals = int(combined_weight.numel())
    total_storage_bytes = sign_mantissa_bytes + encoded_exp_bytes + split_metadata_bytes
    original_bytes = total_vals * 2

    payload_bits_per_weight = 8.0 + avg_exp_bits
    payload_compression = 16.0 / payload_bits_per_weight
    final_bits_per_weight = (total_storage_bytes * 8) / total_vals
    final_compression = original_bytes / total_storage_bytes

    group_dir.mkdir(parents=True, exist_ok=True)

    # Save sign+mantissa stream
    torch.save(sign_mantissa.cpu(), group_dir / "sign_mantissa.pt")

    # Save encoded exponent stream
    with open(group_dir / "encoded_exponents.bin", "wb") as f:
        f.write(encoded)

    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "group_name": f"{shard}/{family}",
        "shard": shard,
        "family": family,
        "num_layers": len(group_layers),
        "total_vals": total_vals,
        "unique_exponents": len(freq_dict),
        "avg_exp_bits": avg_exp_bits,
        "payload_bits_per_weight": payload_bits_per_weight,
        "payload_compression": payload_compression,
        "sign_mantissa_bytes": sign_mantissa_bytes,
        "encoded_exp_bytes": encoded_exp_bytes,
        "split_metadata_bytes": split_metadata_bytes,
        "total_storage_bytes": total_storage_bytes,
        "original_bytes": original_bytes,
        "final_bits_per_weight": final_bits_per_weight,
        "final_compression": final_compression,
        "split_positions": split_positions,
        "layers": [
            {
                "name": x["name"],
                "type": x["type"],
                "shape": x["shape"],
                "num_vals": x["num_vals"],
            }
            for x in group_layers
        ],
        "exponent_frequencies": {str(k): int(v) for k, v in freq_dict.items()},
    }

    with open(group_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    grouped = build_grouped_layers(MODEL_NAME)

    if not grouped:
        print("No groups found.")
        return

    print(f"Found {len(grouped)} grouped configurations")
    print(f"Exporting to: {output_root.resolve()}\n")

    manifest = {
        "model_name": MODEL_NAME,
        "output_dir": str(output_root.resolve()),
        "groups": [],
    }

    total_vals = 0
    total_original_bytes = 0
    total_storage_bytes = 0

    for shard, family in sorted(grouped.keys()):
        group_layers = grouped[(shard, family)]
        group_name = f"{shard}__{family}"
        group_dir = output_root / group_name

        metadata = export_one_group(group_dir, shard, family, group_layers)

        total_vals += metadata["total_vals"]
        total_original_bytes += metadata["original_bytes"]
        total_storage_bytes += metadata["total_storage_bytes"]

        manifest["groups"].append({
            "group_name": metadata["group_name"],
            "group_dir": str(group_dir),
            "num_layers": metadata["num_layers"],
            "total_vals": metadata["total_vals"],
            "final_bits_per_weight": metadata["final_bits_per_weight"],
            "final_compression": metadata["final_compression"],
        })

        print("=" * 72)
        print(f"EXPORTED GROUP: {metadata['group_name']}")
        print(f"  num_layers              = {metadata['num_layers']}")
        print(f"  total_vals              = {metadata['total_vals']}")
        print(f"  avg_exp_bits            = {metadata['avg_exp_bits']:.4f}")
        print(f"  sign_mantissa_bytes     = {metadata['sign_mantissa_bytes']}")
        print(f"  encoded_exp_bytes       = {metadata['encoded_exp_bytes']}")
        print(f"  split_metadata_bytes    = {metadata['split_metadata_bytes']}")
        print(f"  total_storage_bytes     = {metadata['total_storage_bytes']}")
        print(f"  final_bits_per_weight   = {metadata['final_bits_per_weight']:.4f}")
        print(f"  final_compression       = {metadata['final_compression']:.4f}x")
        print(f"  output_dir              = {group_dir}")

    overall = {
        "total_vals": total_vals,
        "original_bytes": total_original_bytes,
        "grouped_storage_bytes": total_storage_bytes,
        "final_bits_per_weight": (total_storage_bytes * 8) / total_vals,
        "final_compression": total_original_bytes / total_storage_bytes,
    }

    manifest["overall_summary"] = overall

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 72)
    print("OVERALL EXPORTED SUMMARY")
    print(f"  total_vals              = {overall['total_vals']}")
    print(f"  original_bytes          = {overall['original_bytes'] / (1024**2):.2f} MB")
    print(f"  grouped_storage_bytes   = {overall['grouped_storage_bytes'] / (1024**2):.2f} MB")
    print(f"  final_bits_per_weight   = {overall['final_bits_per_weight']:.4f}")
    print(f"  final_compression       = {overall['final_compression']:.4f}x")
    print(f"  manifest                = {(output_root / 'manifest.json').resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
