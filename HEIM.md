# heim/main — downstream carries

This branch carries a small set of fixes against `Avarok-Cybersecurity/atlas` for use in the
[heim](https://github.com/tscholak/heim) household voice assistant. It's rebased onto
`upstream/main` when we want to pull in new work; carried commits get squashed into PRs
to upstream as their fixes mature.

## Carried commits

| topic | rationale | upstream PR |
|---|---|---|
| `heim: refuse Q12 kernel-batched prefill on SSM models` | Narrows the eligibility gate of `kernel_batched_eligible` in `crates/spark-model/src/model/trait_impl/prefill_b/batch_kernel.rs`, refusing SSM models. The Q12 Path B kernel-batched orchestration documents itself as "compile-only / hardware validation pending"; on hybrid Mamba-MoE models (qwen3_5_moe) the batched SSM dispatcher mis-indexes `ssm_pool` and poisons the CUDA context. This patch is necessary but **not sufficient** to fix the user-visible crash — full diagnosis in `heim/docs/atlas-bug-mixed-batch-illegal-address.md`. | not yet filed |
| `heim: bump cudarc 0.19.2 -> 0.19.7` | atlas's Cargo.toml already declares `cudarc = "^0.19"`; the upstream Cargo.lock at this rev pins 0.19.2 which panics on CUDA 13.2. cudarc 0.19.5+ adds CUDA 13.2 support. Lock-only change, no source impact. | should be a trivial upstream PR |

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
