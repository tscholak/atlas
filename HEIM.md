# heim/main — downstream carries

This branch carries a small set of fixes against `Avarok-Cybersecurity/atlas` for use in the
[heim](https://github.com/tscholak/heim) household voice assistant. It's rebased onto
`upstream/main` when we want to pull in new work; carried commits get squashed into PRs
to upstream as their fixes mature.

## Carried commits

| topic | rationale | upstream status |
|---|---|---|
| `heim: refuse Q12 kernel-batched prefill on SSM models` | Narrows the eligibility gate of `kernel_batched_eligible` in `crates/spark-model/src/model/trait_impl/prefill_b/batch_kernel.rs`, refusing SSM models. The Q12 Path B kernel-batched orchestration documents itself as "compile-only / hardware validation pending"; structurally-correct narrowing of an untested path, kept as defense-in-depth. | not filed |
| `heim: bump cudarc 0.19.2 -> 0.19.7` | atlas's Cargo.toml already declares `cudarc = "^0.19"`; the upstream Cargo.lock at this rev pins 0.19.2 which panics on CUDA 13.2. cudarc 0.19.5+ adds CUDA 13.2 support. Lock-only change, no source impact. | trivial upstream PR candidate |
| `heim: devenv.sh dev shell for inner-loop iteration` | Adds `devenv.yaml` + `devenv.nix` (standalone-mode) at the fork root mirroring heim's `packages/atlas/default.nix` build env. Lets us iterate on the Spark with incremental `cargo` builds in seconds instead of running the full Nix rebuild for every code change. Self-contained dev tooling; nothing on the binary side. | not for upstream |
| `heim: split decode/prefill scratch into two BufferArenas` | `TransformerModel` now owns `buffers` (primary, decode arena) and `secondary_buffers` (prefill arena). `mixed_forward_batch` routes the prefill side to `secondary_buffers` because its caller's stream differs from `default_stream`; with shared `self.buffers.*` singletons, the two halves race and trigger `CUDA_ERROR_ILLEGAL_ADDRESS` under concurrent load. Disjoint memory eliminates the race. **Correctness-only**: aggregate concurrent throughput is unchanged (~50 t/s for N≥2) because the real bottleneck is ~55 ms/tick of scheduler-side overhead between successive `decode_batch` calls in `spark-server/src/scheduler` — see `heim/docs/atlas-bug-mixed-batch-illegal-address.md`. Concurrent-throughput remediation is a separate effort against the scheduler. | not filed (correctness fix; would land bundled with the scheduler work) |
| `heim: lockprof diagnostic instrumentation` | Adds `tracing::info!(target: "atlas::lockprof", …)` around `self.kv_cache.lock()` in `decode_a2.rs` and `prefill_b.rs`, plus model-call timing in `scheduler/decode_step.rs` and `scheduler/phase_continue_prefills/run_batched_mixed.rs`. Silent at default RUST_LOG; enable with `RUST_LOG="info,atlas::lockprof=info"`. Used to localise the n≥2 concurrent-decode bottleneck (mutex never contended; ~65% of each scheduler tick is non-model work). | not for upstream (heim-specific diagnostic) |
| `heim: vendor xgrammar-rs at heim/main wrapping xgrammar v0.2.1` | `vendor/xgrammar-rs/` mirrors `tscholak/xgrammar-rs` branch `heim/main` (fork of `trymirai/xgrammar-rs`), which bumps the C++ `xgrammar` submodule from v0.1.33 → v0.2.1 (xgrammar-2, May 2026). The wrapper itself needed three changes: `TraverseDraftTree` moved from `xgrammar::` free function to a `GrammarMatcher::TraverseDraftTree` method (returns bool for timeout); serialization version `v11` → `v13`; `test_structural_tag_error` marked `#[ignore]` because xgrammar v0.2.x relaxed some structural-tag analyzer rejections (atlas constructs valid tags only, so the negative test isn't load-bearing). `spark-server/Cargo.toml`'s `xgrammar-rs` constraint bumped `0.1` → `0.2`. Required-parameter enforcement bug from v0.1.x (`tests/qwen3_coder_required.rs`) is verified upstream-fixed in v0.2.1. Source for the xgrammar C++ library is supplied by heim's Nix derivation via `XGRAMMAR_SRC_DIR` (see `vendor/xgrammar-rs/build/submodules.rs`); the `xgrammar/` submodule subdir is intentionally not vendored — local dev sets the env var to a checked-out v0.2.1. | trymirai/xgrammar-rs has no v0.2.x release; once run time accumulates we'll PR the bump upstream |

## Iterating on atlas

The fork carries a `devenv.nix` + `devenv.yaml` (standalone-mode
[devenv.sh](https://devenv.sh)) that mirrors heim's
`packages/atlas/default.nix` build env: same `cudaPackages_13_2`,
same `rust-toolchain.toml` pin, same `XGRAMMAR_SRC_DIR` rev, same
`ATLAS_TARGET_*` env vars. On the Spark:

```sh
git clone git@github.com:tscholak/atlas.git ~/atlas-dev
cd ~/atlas-dev
git checkout heim/main
devenv shell                            # enters the dev env
cargo build --release -p spark-server   # ~10-15 min first build,
                                        # seconds to minutes incremental
```

The resulting `./target/release/atlas` is functionally identical to
the binary heim's Nix derivation produces. Heim's
`dev/atlas-iterate.sh` wraps the rsync + cargo + binary-swap + smoke
test sequence so each iteration is one command.

For production deploys, the Nix derivation in heim is still the
source of truth; the dev shell only exists to compress the inner loop.
See heim's `docs/atlas-bug-mixed-batch-illegal-address.md` for the
broader workflow context.

## How heim consumes this branch

heim's `packages/atlas/default.nix` uses `fetchFromGitHub { owner = "tscholak"; repo = "atlas"; rev = "<heim/main tip>"; }`. To advance the closure to a new atlas commit on this branch:

```sh
# In ~/Projects/atlas:
git push origin heim/main

# In heim:
nix-prefetch-github tscholak atlas --rev <new-sha>
# update rev + hash in packages/atlas/default.nix
make deploy SPARK_IP=<spark-lan-ip>
```

## Pulling upstream work

```sh
cd ~/Projects/atlas
git fetch upstream
git rebase upstream/main heim/main
git push --force-with-lease origin heim/main
```

## Upstreaming a carried commit

```sh
cd ~/Projects/atlas
git checkout -b <topic> upstream/main
git cherry-pick <heim/main commit>
git push origin <topic>
gh pr create --repo Avarok-Cybersecurity/atlas --base main --head tscholak:<topic>
```

Once accepted upstream, the next rebase will drop the carry from `heim/main` automatically.
