import torch
from dahuffman import HuffmanCodec
from transformers import AutoModelForCausalLM


def bf16_to_int16(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16:
        raise ValueError(f"Expected bfloat16 tensor, got {x.dtype}")
    return x.view(torch.int16)


def split_bf16_bits(x: torch.Tensor):
    """
    Split BF16 values into:
    - exponent: 8 bits
    - sign + mantissa: 8 bits
    """
    w = bf16_to_int16(x)
    exponent = ((w >> 7) & 0xFF).to(torch.uint8)
    sign_mantissa = (((w >> 8) & 0x80) | (w & 0x7F)).to(torch.uint8)
    return w, exponent, sign_mantissa


def reconstruct_bf16(exp: torch.Tensor, sm: torch.Tensor):
    """
    Reconstruct BF16 values from exponent and sign+mantissa streams.
    """
    exp = exp.to(torch.int16)
    sm = sm.to(torch.int16)

    sign = (sm & 0x80) << 8
    mantissa = sm & 0x7F
    bits = sign | (exp << 7) | mantissa

    return bits, bits.view(torch.bfloat16)


def build_codec(exponent: torch.Tensor):
    """
    Build a Huffman codec for a given exponent tensor.
    """
    vals, freqs = torch.unique(exponent, return_counts=True)
    freq_dict = {int(v): int(f) for v, f in zip(vals.tolist(), freqs.tolist())}
    codec = HuffmanCodec.from_frequencies(freq_dict)
    return codec, freq_dict


def avg_bits(codec, freq_dict) -> float:
    """
    Compute average number of bits per exponent symbol under the Huffman code.
    """
    table = codec.get_code_table()
    total = sum(freq_dict.values())
    return sum(
        length * freq_dict[symbol] / total
        for symbol, (length, _) in table.items()
        if symbol in freq_dict
    )


def encode_exponents(exponent: torch.Tensor, codec):
    return codec.encode(exponent.flatten().tolist())


def decode_exponents(encoded, codec, shape):
    decoded = codec.decode(encoded)
    decoded = torch.tensor(decoded, dtype=torch.uint8).reshape(shape)
    return decoded


def load_model(name: str):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model


def get_layer_type(name: str) -> str:
    """
    Group layers by functional type for analysis.
    """
    for k in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]:
        if name.endswith(k):
            return k
    return "other"


def get_linear_layers(model):
    """
    Return all BF16 linear layer weights as (name, weight) pairs.
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.detach().cpu()
            if w.dtype == torch.bfloat16:
                layers.append((name, w))
    return layers