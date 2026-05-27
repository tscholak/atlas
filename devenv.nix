{ pkgs, lib, ... }:

# Re-import nixpkgs with `config.allowUnfree = true` because devenv's
# `nixpkgs.config.allowUnfree` setting in `devenv.yaml` does not, in
# practice, propagate to the cuda EULA check on aarch64-linux through
# the `dgx-spark`-overlay-on-top-of-nixpkgs input chain (tested with
# every yaml form documented and several undocumented variants;
# cuda_nvcc still hits the unfree gate). The standard nixpkgs cuda
# pattern is `import nixpkgs { config.allowUnfree = true; ... }`, so we
# match that here at the module level.
#
# The trade-off: we lose dgx-spark.overlays.fixes for the dev shell,
# meaning the dev binary's cudaPackages are vanilla nixpkgs's
# (cudaPackages_13_2 if available, else whatever's pinned). For an
# inner-loop dev binary that only needs to be functionally equivalent
# — not bit-identical — to the production binary, this is acceptable;
# production deploys still go through heim's Nix derivation which
# applies the full dgx-spark overlay. Any ABI mismatch will surface
# at the verification step (single-request output diff vs current
# pinned fork tip).

# Dev shell for iterating on atlas itself.
#
# Mirrors the build env in heim's `packages/atlas/default.nix` so that
# `cargo build --release -p spark-server` in this shell produces a
# binary functionally identical to the one Nix builds for production
# deploys. Subsequent incremental builds reuse the local `target/`
# directory; first build is ~10–15 min, edit-to-build is seconds.
#
# Workflow (per heim/main carries):
#   1. `devenv shell` here (or direnv via .envrc).
#   2. `cargo build --release -p spark-server`.
#   3. Test by stopping atlas-qwen.service and running
#      `./target/release/atlas serve …` directly. See `HEIM.md` and
#      heim's `dev/atlas-iterate.sh` for the full inner loop.

let
  # Re-import nixpkgs with explicit allowUnfree (see file docstring).
  cudaPkgs = import pkgs.path {
    inherit (pkgs) system;
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  };

  # Pin cudaPackages_13_2 explicitly — production atlas builds against
  # cudaPackages_13_2 (via dgx-spark overlay's cudaPackages alias).
  # Default `cudaPackages` in the unstable nixpkgs we're following is
  # 12.9; that would produce libcudart.so.12 link refs, which fail at
  # runtime on the Spark whose driver provides libcudart.so.13.
  cuda13 = cudaPkgs.cudaPackages_13_2;

  # cudarc's build.rs searches `$CUDA_HOME/lib64` (x86 layout). nixpkgs
  # aarch64 places everything under `lib`. Symlink to make both work.
  # Pattern lifted from heim's packages/atlas/default.nix.
  cudaToolkit = cudaPkgs.symlinkJoin {
    name = "atlas-cuda-toolkit";
    paths = with cuda13; [
      cuda_nvcc
      cuda_cudart
      cuda_cccl
      (lib.getDev cuda_cudart)
      (lib.getDev cuda_cccl)
    ];
    postBuild = ''
      ln -s lib $out/lib64
    '';
  };

  # xgrammar source for xgrammar-rs's build.rs. Avoids the in-build
  # git fetch that fails offline. Same rev as heim's atlas derivation,
  # so the dev binary's xgrammar matches production bit-for-bit.
  xgrammarSrc = pkgs.fetchFromGitHub {
    owner = "mlc-ai";
    repo = "xgrammar";
    rev = "v0.1.32";
    hash = "sha256-TIWuMI4d3ETc/ZItYwVmLtSHmQczevrXXHEAikg9Tmw=";
    fetchSubmodules = true;
  };
in
{
  # --- Languages ---

  # rust-toolchain.toml at repo root pins channel = "1.93.1" +
  # rustfmt + clippy components; devenv's languages.rust module
  # reads it automatically.
  languages.rust.enable = true;

  # --- Packages ---

  packages = (with pkgs; [
    pkg-config
    cmake
    autoAddDriverRunpath
    # bindgen for the sys crates (cudarc, xgrammar-rs) needs libclang.
    # `rustPlatform.bindgenHook` in heim's derivation expands to
    # LIBCLANG_PATH + the runtime; we set LIBCLANG_PATH below.
    llvmPackages.libclang
  ]) ++ (with cuda13; [
    cuda_nvcc
    cuda_cudart
    nccl
  ]);

  # --- Environment ---
  #
  # Must match heim's packages/atlas/default.nix `env` block 1:1 so
  # the dev binary and production binary build against the same
  # compile-time configuration (target hardware, target model, CUDA
  # search path, xgrammar source).

  env = {
    CUDA_HOME = "${cudaToolkit}";
    ATLAS_TARGET_HW = "gb10";
    ATLAS_TARGET_MODEL = "qwen3.6-35b-a3b";
    ATLAS_TARGET_QUANT = "*";
    LIBCLANG_PATH = "${lib.getLib pkgs.llvmPackages.libclang}/lib";
    XGRAMMAR_SRC_DIR = "${xgrammarSrc}";
  };

  # --- Shell ---

  enterShell = ''
    echo "atlas dev shell (heim/main carry workflow)"
    echo "  cargo build --release -p spark-server     -> ./target/release/atlas"
    echo "  See HEIM.md for the inner-loop iteration script."
  '';
}
