# ribbonGo

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A high-performance **Ribbon filter** implementation in Go — the space-efficient probabilistic data structure that is **practically smaller than Bloom and Xor filters**.

Based on the paper:

> **"Ribbon filter: practically smaller than Bloom and Xor"**
> Peter C. Dillinger & Stefan Walzer, 2021
> ([arXiv:2103.02515](https://arxiv.org/abs/2103.02515))

## What is a Ribbon Filter?

A **Ribbon filter** is a static, space-efficient probabilistic data structure for approximate set membership queries — the same problem Bloom filters solve, but using significantly less space.

Given a set of keys, a Ribbon filter can answer *"is this key in the set?"* with:

- **No false negatives** — if a key is in the set, the filter always says yes.
- **Configurable false positive rate** — non-member keys have a tunable probability of being incorrectly reported as members (FPR ≈ 2<sup>−r</sup> for r result bits).
- **Near-optimal space** — approaches the information-theoretic lower bound, using only ~1–2% more space than the minimum possible.

### Ribbon vs Bloom vs Xor

| Property | Bloom | Xor | **Ribbon** |
|----------|-------|-----|------------|
| Space overhead | ~44% over optimal | ~23% over optimal | **~1–5%** over optimal |
| Construction | Fast, incremental | Fast, static | Moderate, static |
| Supports deletion | No (without counting) | No | No |
| FPR at 1% target (bits/key) | 9.6 | 9.8 | **7.2** |

Ribbon filters are ideal for **read-heavy, write-once** workloads — LSM-tree engines (RocksDB, LevelDB), static lookup tables, networking data planes, and any scenario where a set is built once and queried millions of times.

## Features

- **Paper-faithful implementation** — every design decision traces to a specific section of Dillinger & Walzer (2021), with `§N` citations throughout the code
- **Configurable ribbon width** — w ∈ {32, 64, 128} to trade construction speed for space efficiency
- **Configurable result bits** — r bits per key for FPR ≈ 2<sup>−r</sup>
- **firstCoeffAlwaysOne optimisation** — deterministic pivot from paper §4, skipping leading-zero scan
- **SoA memory layout** — Struct-of-Arrays for maximum cache utilisation (8 coefficients per cache line at w=64)
- **Width-specialised inner loops** — pure `uint64` path for w≤64, separate lo/hi ops for w=128
- **Software-pipelined prefetching** — `AddRange` hides L2/L3 latency by prefetching the next key's slot
- **Zero allocations on hot paths** — no heap escapes during construction or query
- **Two-phase hash pipeline** — hash keys once, remix per seed attempt; uses [XXH3](https://github.com/zeebo/xxh3) (64-bit)
- **Research-friendly** — all paper parameters exposed; reference (`slow*`) implementations included for cross-validation

## Installation

```bash
go get github.com/ribnonGo/ribbon
```

Requires **Go 1.25+**.

## Project Status

> **🚧 Under active development** — the core pipeline is being built layer by layer.

| Layer | File | Status | Description |
|-------|------|--------|-------------|
| uint128 | `uint128.go` | ✅ Complete | 128-bit integer for coefficient rows |
| Hash | `hash.go` | ✅ Complete | Two-phase hash pipeline (key → hash → derive) |
| **Bander** | `bander.go` | ✅ Complete | Gaussian elimination / matrix construction |
| Back-substitution | `backsubst.go` | 🔜 Next | Solve the banded system for the solution vector |
| Query | `query.go` | ⬜ Planned | Membership query |
| Filter | `filter.go` | ⬜ Planned | Public API: Build / Query / Serialize |

The public `Build()` and `Query()` API will be available once the full pipeline is complete. The internal layers are stable and extensively tested.

## Architecture

The implementation follows the paper's algorithmic pipeline:

```
Key → [Hash] → [Bander] → [BackSubst] → [Query]
       §2         §2,§4       §2           §2
```

**Hash layer** (`hash.go`) — two-phase pipeline: hash each key once with XXH3, then cheaply remix per seed attempt to derive `(start, coeffRow, result)` triples.

**Bander layer** (`bander.go`) — on-the-fly Gaussian elimination over GF(2). Converts hashed key equations into an upper-triangular banded matrix. This is the hottest code path during construction.

**Back-substitution** (`backsubst.go`) — solves the upper-triangular system to produce the compact solution vector that encodes the filter.

**Query** (`query.go`) — evaluates a key against the solution vector. One hash, one dot product over GF(2).

### Key Optimisations

**SoA layout** — coefficient data is stored in parallel arrays (`coeffLo []uint64`, `coeffHi []uint64`, `result []uint8`) instead of an array of structs. For w≤64, `coeffHi` is `nil` — zero memory, zero operations. This doubles the number of coefficients per cache line compared to the naive AoS layout.

**Width specialisation** — `Add()` dispatches to `addW64()` (pure `uint64`, ~10 ARM64 instructions per elimination step) or `addW128()` (separate lo/hi `uint64` ops). The generic `uint128.rsh()` with its 4-branch dispatch is avoided entirely.

**Software-pipelined prefetching** — `AddRange()` touches `coeffLo[nextKey.start]` while processing the current key, pulling the next cache line into L1. This mirrors RocksDB's `BandingAddRange` approach and yields 20–26% throughput improvement at scale.

## Benchmarks

**Apple M3 Pro, Go 1.25, ARM64**

### Per-key `Add` (low load, amortised)

| Width | firstCoeffAlwaysOne | ns/op | Throughput |
|-------|---------------------|-------|------------|
| w=32 | true | 4.99 | ~200M keys/sec |
| w=64 | true | 4.99 | ~200M keys/sec |
| w=128 | true | 6.67 | ~150M keys/sec |
| w=32 | false | 5.71 | ~175M keys/sec |
| w=64 | false | 5.73 | ~175M keys/sec |
| w=128 | false | 7.33 | ~137M keys/sec |

### Full banding pass: `Add`-loop vs `AddRange` (with prefetching)

| Keys | Width | Add-loop | AddRange | Speedup |
|------|-------|----------|----------|---------|
| 10K | w=64 | 71.6 µs (~140M keys/sec) | 52.7 µs (~190M keys/sec) | **1.36×** |
| 10K | w=128 | 110.2 µs (~91M keys/sec) | 94.6 µs (~106M keys/sec) | **1.16×** |
| 100K | w=64 | 1.84 ms (~55M keys/sec) | 1.47 ms (~68M keys/sec) | **1.25×** |
| 100K | w=128 | 2.18 ms (~46M keys/sec) | 1.95 ms (~51M keys/sec) | **1.12×** |

All benchmarks report **zero allocations** on hot paths.

### Run benchmarks yourself

```bash
go test -bench=. -benchmem -count=5 ./...
```

## Testing

162 subtests across 21 test functions — all passing.

Tests include:
- **Correctness**: single insertion, collision chains, 128-bit boundary crossing, redundant/contradictory equations
- **Cross-validation**: optimised `Add()` vs reference `slowAdd()`, `AddRange()` vs `Add()`-loop, `AddRange()` vs `slowAddRange()` — all verified slot-by-slot across 6 configurations (3 widths × 2 fcao settings)
- **Statistical**: success rate validation over 50 seeds × 5000 keys
- **Edge cases**: single-slot bander, empty ranges, failure-stops-early

```bash
go test -v -count=1 ./...
```

## Contributing

Contributions are welcome! This project follows the paper's design closely — please read the relevant paper section before modifying any algorithm.

### Getting started

```bash
git clone https://github.com/RibbonGo/ribbonGo.git
cd ribbonGo
go test ./...
```

### Guidelines

- **Paper-first**: every design decision should cite a specific section (`§N`) of Dillinger & Walzer (2021)
- **RocksDB cross-references**: use `[RocksDB: FunctionName in file.h]` format for implementation parallels
- **All 6 configs**: tests must cover w ∈ {32, 64, 128} × firstCoeffAlwaysOne ∈ {true, false}
- **Reference implementations**: include `slow*` variants for cross-validation of optimised code
- **Zero allocations**: hot paths must not escape to heap — verify with `go test -bench=X -benchmem`
- **Naming**: unexported everything (`standardBander`, not `StandardBander`); constants use `kCamelCase`

### What's next

The remaining layers to implement, in order:

1. **Back-substitution** (`backsubst.go`) — solve the upper-triangular system
2. **Query** (`query.go`) — membership test against the solution vector
3. **Filter** (`filter.go`) — public API: `Build`, `Query`, `Serialize`/`Deserialize`

## References

- Dillinger, P. C. & Walzer, S. (2021). *Ribbon filter: practically smaller than Bloom and Xor*. [arXiv:2103.02515](https://arxiv.org/abs/2103.02515)
- RocksDB Ribbon filter implementation: [ribbon_impl.h](https://github.com/facebook/rocksdb/blob/main/util/ribbon_impl.h), [ribbon_alg.h](https://github.com/facebook/rocksdb/blob/main/util/ribbon_alg.h)
- XXH3 hash function (Go): [github.com/zeebo/xxh3](https://github.com/zeebo/xxh3)

## License

[MIT](LICENSE) © 2026 RibbonGo
