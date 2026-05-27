# heim/main — downstream carries

This branch carries a small set of fixes against `Avarok-Cybersecurity/atlas` for use in the
[heim](https://github.com/tscholak/heim) household voice assistant. It's rebased onto
`upstream/main` when we want to pull in new work; carried commits get squashed into PRs
to upstream as their fixes mature.

## Carried commits

| topic | rationale | upstream PR |
|---|---|---|
| `heim: refuse Q12 kernel-batched prefill on SSM models` | Narrows the eligibility gate of `kernel_batched_eligible` in `crates/spark-model/src/model/trait_impl/prefill_b/batch_kernel.rs`, refusing SSM models. The Q12 Path B kernel-batched orchestration documents itself as "compile-only / hardware validation pending"; structurally-correct narrowing of an untested path, kept as defense-in-depth even though it didn't fix the user-visible crash by itself. | not yet filed |
| `heim: bump cudarc 0.19.2 -> 0.19.7` | atlas's Cargo.toml already declares `cudarc = "^0.19"`; the upstream Cargo.lock at this rev pins 0.19.2 which panics on CUDA 13.2. cudarc 0.19.5+ adds CUDA 13.2 support. Lock-only change, no source impact. | should be a trivial upstream PR |
| `heim: sync default_stream at decode_batch_dispatch exit (cross-stream race fix)` | `decode_batch_dispatch` (n>=2 non-EP path) silently shadows the caller's stream with `default_stream` for graph-capture determinism, but `prefill_chunk_dispatch` respects the caller's stream. The default `mixed_forward_batch` impl calls them back-to-back assuming they share a stream → two streams writing the same `self.buffers.*` singletons → `CUDA_ERROR_ILLEGAL_ADDRESS`. Initial fix: CPU `gpu.synchronize(default_stream)` at function exit. **Superseded by the event-handshake commit below.** | filed as part of the follow-up PR |
| `heim: cross-stream event handshake at decode_batch_dispatch exit` | Replaces the CPU-sync fix above. Adds a dedicated `decode_batch_done_event` field on `TransformerModel`; records on default_stream and has the caller's stream wait. GPU-side ordering, no CPU stall, lets decode and prefill overlap on the GPU. Aggregate concurrent throughput climbs from ~41 → expected ~150–200 tok/s on Qwen3.6-35B-A3B-FP8. Full diagnosis in `heim/docs/atlas-bug-mixed-batch-illegal-address.md`. | not yet filed |

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
