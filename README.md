# dfloat_tenstorrent


### Project Structure:

- experiments/
  - coding_compare.py: compression analysis
  - grouped_format.py: creates grouped artifacts
  - execution.py: reload + execute grouped artifacts
  - grouped_runtime.py: grouped validation

- core/
  - utils.py: shared helper functions

- artifacts/
  - grouped compressed model format (use for execution)


### Compression Results

| Strategy                  | Bits / Weight | Compression |
|--------------------------|--------------|-------------|
| Per-layer                | 10.6519      | 1.5021x     |
| Per-type                 | 10.6559      | 1.5015x     |
| Global                   | 10.6723      | 1.4992x     |
| Per-shard (naive)        | 10.6679      | 1.4998x     |
| Per-shard (type-aware)   | 10.6572      | 1.5013x     |


Analysis:
- Per-layer gives the best compression but is not hardware-friendly
- Per-type performs almost as well as per-layer
- Global encoding is slightly worse
- Naive per-shard grouping degrades compression
- Per-(shard, family) recovers most of the loss and is the best hardware-aware choice

 

 
