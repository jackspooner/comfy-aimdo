# Linux CUDA regression evidence

On 2026-10-09, RTX 5090 / PyTorch 2.14.1+cu130 / Python 3.13.16:

- Installed upstream 0.5.5 (source tag commit
  `3b8e8c162efeb9470d912609a7a6e7a2b1c693ec`, binary build ID
  `103edc8e96592bac75ed026824daae81fe8ad9a9`) reproducibly crashed on the
  second root handoff of retained aliased allocations. GDB identified
  `physical_page_unref(page=0x0)`, called by `finalize_rogues` line 739 and
  `malloc_graph_pop` line 1177, matching the production crash signature.
- The fork regression passes repeated handoff, replay, abort/destruction,
  tensor-content checks and return to baseline native VRAM usage.
- Replacement allocation and mapping fault injection reports compile errors
  while preserving escaped tensor contents through graph destruction. These
  tests use separate processes: existing failed-handoff lifecycle behavior does
  not certify recovery/reuse of a failed graph in the same process.
- 42 of the original 45-test run (44 upstream plus the new repeated-handoff
  test) passed. Three upstream assertions also fail against the original
  installed wheel: `malloc-graph-different-stream-cuda.py`,
  `malloc-graph-aliasing-cuda.py`, and `malloc-graph-assert-break-cuda.py`.
  They concern off-stream replay, peak accounting and allocator error retry.
  They are retained, not weakened or silently skipped. The separate fault
  injection test subsequently passed and is included in the suite.

These checks prove the isolated crash regression on this CUDA configuration.
They do not establish live H3/Omni changeover acceptance, other CUDA versions,
HIP or Windows. Record actual deployment and model acceptance separately.
