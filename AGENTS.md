# Development

Run `python3 scripts/setup_worktree.py` in each Git CLI task worktree. Linux
requires Python 3.12+, GCC, Make and CMake. Provision funchook 1.1.3 through
Download Manager 2 and set `AIMDO_FUNCHOOK_ARCHIVE` to its archive, or place it in
`build/funchook-1.1.3.tar.gz`. Setup checks its SHA256 and builds checkout-local
CUDA/HIP libraries without downloading, touching a service or starting inference.
It is safe to repeat. CUDA tests require an explicitly prepared PyTorch runtime
and an idle, coordinated GPU; run `PYTHONPATH=. <python> tests/run-malloc-graph-cuda.py`.

After preparation use managed GitNexus ensure for this exact checkout with
embeddings disabled. Maintain reusable guidance under `skills/`; register only
the integrated canonical folder using `skills register`, then `skills reconcile`
and `skills doctor`. Preserve upstream attribution and license.
