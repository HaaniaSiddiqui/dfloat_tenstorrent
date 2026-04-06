"""
Compare different exponent-coding granularities.

Schemes:
- per_layer: one Huffman code per analyzed weight matrix / linear layer
- per_type: one Huffman code shared across all layers of the same type
- global: one Huffman code shared across all analyzed linear weights
- per_shard_naive: one Huffman code per contiguous hardware-style shard
- per_shard_type_aware: one Huffman code per (shard, coarse layer family)

Important:
- This script compares payload-only exponent coding
- It does NOT include LUT/gap/output-position overhead
"""

import os
import sys
import math
import torch
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.utils import (
    build_codec,
    avg_bits,
    load_model,
    get_layer_type,
    get_linear_layers,
    split_bf16_bits,
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


def get_coarse_family(layer_type: str) -> str:
    if layer_type in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attn"
    if layer_type in {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    if layer_type == "lm_head":
        return "lm_head"
    return "other"


def load_layer_exponents(model_name=MODEL_NAME):
    model = load_model(model_name)
    linear_layers = get_linear_layers(model)

    layers = []
    for name, weight in linear_layers:
        _, exponent, sign_mantissa = split_bf16_bits(weight)
        layer_type = get_layer_type(name)
        layers.append({
            "name": name,
            "type": layer_type,
            "family": get_coarse_family(layer_type),
            "num_vals": weight.numel(),
            "exponent": exponent.flatten(),
            "sign_mantissa": sign_mantissa.flatten(),
            "shard": assign_shard(name),
        })
    return layers


def print_scheme_summary(name, total_weights, total_original_bytes, total_encoded_exp_bytes):
    total_sign_mantissa_bytes = total_weights
    total_compressed_bytes = total_sign_mantissa_bytes + total_encoded_exp_bytes

    bits_per_weight = (total_compressed_bytes * 8) / total_weights
    compression_ratio = total_original_bytes / total_compressed_bytes

    print("=" * 60)
    print(name.upper())
    print(f"Total weights               : {total_weights}")
    print(f"Original bytes              : {total_original_bytes / (1024**2):.2f} MB")
    print(f"Sign+mantissa bytes         : {total_sign_mantissa_bytes / (1024**2):.2f} MB")
    print(f"Encoded exponent bytes      : {total_encoded_exp_bytes / (1024**2):.2f} MB")
    print(f"Bits per weight             : {bits_per_weight:.4f}")
    print(f"Compression ratio           : {compression_ratio:.4f}x")
    print("=" * 60)


def evaluate_per_layer(layers):
    total_weights = 0
    total_original_bytes = 0
    total_encoded_exp_bytes = 0

    for layer in layers:
        codec, freq_dict = build_codec(layer["exponent"])
        avg_exp_bits = avg_bits(codec, freq_dict)

        total_weights += layer["num_vals"]
        total_original_bytes += layer["num_vals"] * 2
        total_encoded_exp_bytes += math.ceil(layer["num_vals"] * avg_exp_bits / 8.0)

    print_scheme_summary("per_layer", total_weights, total_original_bytes, total_encoded_exp_bytes)


def evaluate_per_type(layers):
    grouped = defaultdict(list)
    for layer in layers:
        grouped[layer["type"]].append(layer)

    total_weights = 0
    total_original_bytes = 0
    total_encoded_exp_bytes = 0

    print("\nPer-type code stats:")
    for layer_type, group in grouped.items():
        concat_exp = torch.cat([x["exponent"] for x in group], dim=0)
        codec, freq_dict = build_codec(concat_exp)
        avg_exp_bits = avg_bits(codec, freq_dict)

        group_weights = sum(x["num_vals"] for x in group)
        group_encoded_exp_bytes = math.ceil(group_weights * avg_exp_bits / 8.0)

        total_weights += group_weights
        total_original_bytes += group_weights * 2
        total_encoded_exp_bytes += group_encoded_exp_bytes

        print(
            f"{layer_type:18s} | "
            f"weights={group_weights:<10d} | "
            f"avg_exp_bits={avg_exp_bits:.4f} | "
            f"encoded_exp_MB={group_encoded_exp_bytes / (1024**2):.2f}"
        )

    print_scheme_summary("per_type", total_weights, total_original_bytes, total_encoded_exp_bytes)


def evaluate_global(layers):
    all_exp = torch.cat([x["exponent"] for x in layers], dim=0)
    codec, freq_dict = build_codec(all_exp)
    avg_exp_bits = avg_bits(codec, freq_dict)

    total_weights = sum(x["num_vals"] for x in layers)
    total_original_bytes = total_weights * 2
    total_encoded_exp_bytes = math.ceil(total_weights * avg_exp_bits / 8.0)

    print(f"\nGlobal avg exponent bits: {avg_exp_bits:.4f}")
    print_scheme_summary("global", total_weights, total_original_bytes, total_encoded_exp_bytes)


def evaluate_per_shard_naive(layers):
    grouped = defaultdict(list)
    for layer in layers:
        grouped[layer["shard"]].append(layer)

    total_weights = 0
    total_original_bytes = 0
    total_encoded_exp_bytes = 0

    print("\nPer-shard naive code stats:")
    for shard_name, group in grouped.items():
        concat_exp = torch.cat([x["exponent"] for x in group], dim=0)
        codec, freq_dict = build_codec(concat_exp)
        avg_exp_bits = avg_bits(codec, freq_dict)

        group_weights = sum(x["num_vals"] for x in group)
        group_encoded_exp_bytes = math.ceil(group_weights * avg_exp_bits / 8.0)

        total_weights += group_weights
        total_original_bytes += group_weights * 2
        total_encoded_exp_bytes += group_encoded_exp_bytes

        print(
            f"{shard_name:18s} | "
            f"layers={len(group):<4d} | "
            f"weights={group_weights:<10d} | "
            f"avg_exp_bits={avg_exp_bits:.4f} | "
            f"encoded_exp_MB={group_encoded_exp_bytes / (1024**2):.2f}"
        )

    print_scheme_summary("per_shard_naive", total_weights, total_original_bytes, total_encoded_exp_bytes)


def evaluate_per_shard_type_aware(layers):
    grouped = defaultdict(list)
    for layer in layers:
        key = (layer["shard"], layer["family"])
        grouped[key].append(layer)

    total_weights = 0
    total_original_bytes = 0
    total_encoded_exp_bytes = 0

    print("\nPer-shard type-aware code stats:")
    for (shard_name, family), group in grouped.items():
        concat_exp = torch.cat([x["exponent"] for x in group], dim=0)
        codec, freq_dict = build_codec(concat_exp)
        avg_exp_bits = avg_bits(codec, freq_dict)

        group_weights = sum(x["num_vals"] for x in group)
        group_encoded_exp_bytes = math.ceil(group_weights * avg_exp_bits / 8.0)

        total_weights += group_weights
        total_original_bytes += group_weights * 2
        total_encoded_exp_bytes += group_encoded_exp_bytes

        print(
            f"{shard_name + '/' + family:18s} | "
            f"layers={len(group):<4d} | "
            f"weights={group_weights:<10d} | "
            f"avg_exp_bits={avg_exp_bits:.4f} | "
            f"encoded_exp_MB={group_encoded_exp_bytes / (1024**2):.2f}"
        )

    print_scheme_summary("per_shard_type_aware", total_weights, total_original_bytes, total_encoded_exp_bytes)


def main():
    print(f"Loading model: {MODEL_NAME}")
    layers = load_layer_exponents(MODEL_NAME)

    if not layers:
        print("No BF16 linear layers found.")
        return

    print(f"Linear layers found: {len(layers)}")

    evaluate_per_layer(layers)
    evaluate_per_type(layers)
    evaluate_global(layers)
    evaluate_per_shard_naive(layers)
    evaluate_per_shard_type_aware(layers)


if __name__ == "__main__":
    main()