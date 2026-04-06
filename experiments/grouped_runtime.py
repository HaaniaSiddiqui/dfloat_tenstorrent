"""
Grouped runtime simulation + grouped storage format definition.

This script does two things together:

1. GROUPED RUNTIME SIMULATION
   - groups layers by (shard, family)
   - compresses exponent stream once per group
   - decodes and reconstructs grouped weights
   - splits them back into original tensors
   - verifies matmul outputs match original tensors exactly

2. GROUPED STORAGE FORMAT DEFINITION
   - reports what each group would need to store:
       * sign+mantissa bytes
       * encoded exponent bytes
       * split metadata
       * layer names / shapes
   - this is a prototype for a shard-aware compressed representation

Current grouping:
- shard: fake hardware partition
- family: attn / mlp / lm_head
"""

import os
import sys
import math
import torch
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.utils import (
    split_bf16_bits,
    reconstruct_bf16,
    build_codec,
    avg_bits,
    encode_exponents,
    decode_exponents,
    load_model,
    get_linear_layers,
    get_layer_type,
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


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
            "shape": tuple(weight.shape),
            "num_vals": weight.numel(),
            "weight": weight,
        })

    return grouped


def build_group_storage_and_reconstruct(group_layers):
    """
    Build grouped compressed representation and reconstruct it back.

    Returns:
        storage_info: metadata / storage summary
        reconstructed_by_layer: dict layer_name -> reconstructed weight tensor
        original_by_layer: dict layer_name -> original weight tensor
    """
    # Concatenate flattened weights
    flat_weights = [x["weight"].reshape(-1) for x in group_layers]
    combined_weight = torch.cat(flat_weights, dim=0)

    original_bits, exponent, sign_mantissa = split_bf16_bits(combined_weight)

    codec, freq_dict = build_codec(exponent)
    avg_exp_bits = avg_bits(codec, freq_dict)

    encoded = encode_exponents(exponent, codec)
    decoded_exponent = decode_exponents(encoded, codec, exponent.shape)

    if not torch.equal(exponent.cpu(), decoded_exponent.cpu()):
        raise RuntimeError("Grouped exponent decode mismatch")

    reconstructed_bits, reconstructed_weight = reconstruct_bf16(
        decoded_exponent,
        sign_mantissa.cpu()
    )

    if not torch.equal(original_bits.cpu(), reconstructed_bits.cpu()):
        raise RuntimeError("Grouped bitwise reconstruction failed")

    if not torch.equal(combined_weight.cpu(), reconstructed_weight.cpu()):
        raise RuntimeError("Grouped tensor reconstruction failed")

    # Split back to original layers
    reconstructed_by_layer = {}
    original_by_layer = {}

    start = 0
    split_sizes = []

    for item in group_layers:
        n = item["num_vals"]
        split_sizes.append(n)

        rec = reconstructed_weight[start:start + n].reshape(item["shape"])
        orig = item["weight"]

        reconstructed_by_layer[item["name"]] = rec
        original_by_layer[item["name"]] = orig

        start += n

    # Storage format summary
    total_vals = combined_weight.numel()
    sign_mantissa_bytes = sign_mantissa.numel()
    encoded_exp_bytes = len(encoded)

    # simple split metadata: store cumulative end positions as int64
    split_positions = []
    running = 0
    for n in split_sizes[:-1]:
        running += n
        split_positions.append(running)

    split_metadata_bytes = len(split_positions) * 8

    payload_bits_per_weight = 8.0 + avg_exp_bits
    payload_compression = 16.0 / payload_bits_per_weight

    total_storage_bytes = sign_mantissa_bytes + encoded_exp_bytes + split_metadata_bytes
    final_bits_per_weight = (total_storage_bytes * 8) / total_vals
    final_compression = (total_vals * 2) / total_storage_bytes

    storage_info = {
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
        "final_bits_per_weight": final_bits_per_weight,
        "final_compression": final_compression,
        "layer_names": [x["name"] for x in group_layers],
        "layer_shapes": [x["shape"] for x in group_layers],
        "split_positions": split_positions,
    }

    return storage_info, reconstructed_by_layer, original_by_layer


def run_grouped_runtime_check(group_name, reconstructed_by_layer, original_by_layer, max_layers_to_test=3):
    """
    For a few layers in the group:
    - run matmul with original weights
    - run matmul with reconstructed weights
    - check exact equality
    """
    checked = []

    layer_names = list(original_by_layer.keys())[:max_layers_to_test]

    for name in layer_names:
        w_orig = original_by_layer[name]
        w_rec = reconstructed_by_layer[name]

        out_features, in_features = w_orig.shape
        x = torch.randn(4, in_features, dtype=torch.bfloat16)

        y_orig = x @ w_orig.t()
        y_rec = x @ w_rec.t()

        exact = torch.equal(y_orig, y_rec)
        close = torch.allclose(y_orig.float(), y_rec.float(), atol=0, rtol=0)

        if not exact:
            raise RuntimeError(f"Matmul exact mismatch in group {group_name}, layer {name}")

        checked.append({
            "layer_name": name,
            "matmul_exact_match": exact,
            "matmul_close_match": close,
        })

    return checked


def main():
    print(f"Loading model: {MODEL_NAME}")
    grouped = build_grouped_layers(MODEL_NAME)

    if not grouped:
        print("No groups found.")
        return

    print(f"Grouped configurations found: {len(grouped)}\n")

    overall_vals = 0
    overall_storage_bytes = 0
    overall_original_bytes = 0

    for group_key in sorted(grouped.keys()):
        shard, family = group_key
        group_name = f"{shard}/{family}"
        group_layers = grouped[group_key]

        storage_info, reconstructed_by_layer, original_by_layer = build_group_storage_and_reconstruct(group_layers)
        runtime_checks = run_grouped_runtime_check(group_name, reconstructed_by_layer, original_by_layer)

        overall_vals += storage_info["total_vals"]
        overall_storage_bytes += storage_info["total_storage_bytes"]
        overall_original_bytes += storage_info["total_vals"] * 2

        print("=" * 72)
        print(f"GROUP: {group_name}")
        print(f"  num_layers              = {storage_info['num_layers']}")
        print(f"  total_vals              = {storage_info['total_vals']}")
        print(f"  unique_exponents        = {storage_info['unique_exponents']}")
        print(f"  avg_exp_bits            = {storage_info['avg_exp_bits']:.4f}")
        print(f"  payload_bits/weight     = {storage_info['payload_bits_per_weight']:.4f}")
        print(f"  payload_compression     = {storage_info['payload_compression']:.4f}x")
        print()
        print("  Storage format:")
        print(f"    sign_mantissa_bytes   = {storage_info['sign_mantissa_bytes']}")
        print(f"    encoded_exp_bytes     = {storage_info['encoded_exp_bytes']}")
        print(f"    split_metadata_bytes  = {storage_info['split_metadata_bytes']}")
        print(f"    total_storage_bytes   = {storage_info['total_storage_bytes']}")
        print(f"    final_bits/weight     = {storage_info['final_bits_per_weight']:.4f}")
        print(f"    final_compression     = {storage_info['final_compression']:.4f}x")
        print()
        print("  Runtime checks:")
        for rc in runtime_checks:
            print(
                f"    {rc['layer_name']}\n"
                f"      matmul_exact_match = {rc['matmul_exact_match']}\n"
                f"      matmul_close_match = {rc['matmul_close_match']}"
            )

    print("\n" + "=" * 72)
    print("OVERALL GROUPED SUMMARY")
    print(f"  total_vals              = {overall_vals}")
    print(f"  original_bytes          = {overall_original_bytes / (1024**2):.2f} MB")
    print(f"  grouped_storage_bytes   = {overall_storage_bytes / (1024**2):.2f} MB")
    print(f"  final_bits/weight       = {(overall_storage_bytes * 8) / overall_vals:.4f}")
    print(f"  final_compression       = {overall_original_bytes / overall_storage_bytes:.4f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()