# Building a Linux wheel

Build the Python extension as a Linux wheel when the deployment environment
cannot compile Rust or fetch Cargo crates. Run this from a checkout of the
intended commit on a machine with Docker. The explicit platform produces an
x86_64 artifact even when the build machine is Apple Silicon.

```bash
mkdir -p /tmp/switchyard-wheel
docker run --rm --platform linux/amd64 \
  -v "$PWD:/src:ro" \
  -v /tmp/switchyard-wheel:/out \
  -w /src \
  rust:1.96.1-bookworm \
  bash -c '
    set -euo pipefail
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-venv pkg-config
    python3 -m venv /tmp/maturin
    /tmp/maturin/bin/pip install "maturin>=1.9,<2.0"
    export RUSTUP_TOOLCHAIN=1.96.1 CARGO_TARGET_DIR=/out/target
    /tmp/maturin/bin/maturin build --release --compatibility off --out /out/wheels
  '
```

The result must be an x86_64 ABI3 wheel, for example
`nemo_switchyard-0.2.0-cp312-abi3-linux_x86_64.whl`. Verify it before
publishing:

```bash
docker run --rm --platform linux/amd64 \
  -v /tmp/switchyard-wheel/wheels:/wheels:ro \
  python:3.12-slim \
  sh -lc 'pip install /wheels/*.whl && python -c "from switchyard_rust.server import Server"'
```

Rebuild the wheel whenever the Rust sources, `Cargo.lock`, Python bindings, or
Python package metadata change. A wheel replaces runtime compilation; do not
rely on a deployment environment having Cargo's registry cache.
