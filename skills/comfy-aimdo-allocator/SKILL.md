---
name: comfy-aimdo-allocator
description: Build, validate and diagnose the Workspace2 comfy-aimdo native allocation/offloading fork, including escaped allocation ownership and pinned wheel deployment.
---

# Native allocator maintenance

This fork owns the CUDA/HIP allocator library, not ComfyUI model policy or the
RuntimeHost GPU reservation. Preserve enabled compilation/dynamic offloading when
repairing native crashes. Use the existing ComfyUI Linux operations and Cutlery
GPU handoff guides for service activation and reservation recovery.

Prepare a task worktree with `python3 scripts/setup_worktree.py`. It requires
Linux, Python 3.12+, GCC, Make, CMake and the verified funchook 1.1.3 archive at
`AIMDO_FUNCHOOK_ARCHIVE` or `build/funchook-1.1.3.tar.gz`; provision downloads through
Download Manager 2. Setup writes only local build outputs, installs nothing,
starts no workload, and may be repeated. Read the root AGENTS.md for indexing and
shared skill registration. No model downloads are required for allocator tests.

Using an explicitly prepared PyTorch Python and an idle coordinated GPU, run
`PYTHONPATH=. <python> tests/malloc-graph-repeated-rogue-alias-cuda.py` for the
v0.5.5 repeated escaped-alias regression. Run `tests/malloc-graph-rogue-failure-cuda.py`
for allocation/map failure injection and `tests/run-malloc-graph-cuda.py` for the
broader suite. Tests allocate real GPU memory; a native crash terminates the test
process. Preserve its log and build identity; do not rerun a production workflow
blindly. CPU/static checks alone do not certify GPU behavior or other platforms.

The repeated-alias fix preserves holes whose storage has transferred to escaped
tensors, releases only mappings still owned by the graph, and rejects unexpected
missing mappings. It changes no Python API or model settings. Capture exact
source commit, native build ID, version and wheel SHA256 for each deployed build.
Publish the owning source before pinning consumers. Keep the previous wheel for
rollback, deploy only with idle serving workloads, and verify loaded version plus
actual model execution after restart. Never overwrite a live mapped shared library.

A `physical_page_unref(NULL)` stack through `finalize_rogues` matches this
regression only when repeated alias handoff is established. Other CUDA/pinned
memory errors require their own evidence. Record failures against the unmodified
version when distinguishing existing upstream/runtime incompatibilities.
