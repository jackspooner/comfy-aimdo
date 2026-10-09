"""Repeated root handoff must not release an already-unmapped alias.

v0.5.5 crashes in physical_page_unref(NULL) on the second root pop.
No model is needed; graph allocations are tens of MiB.
"""
import gc
import os
os.environ['PYTORCH_ALLOC_CONF'] = 'backend:cudaMallocAsync'
import torch
import comfy_aimdo.control as aimdo

M = 1024 * 1024
assert aimdo.init('cuda')
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device='cuda')
torch.cuda.synchronize()
baseline = aimdo.get_total_vram_usage()

for finish in ('destroy', 'abort'):
    graph = aimdo.record(torch.cuda.current_stream())
    for size in (8, 16):
        graph.push('large')
        value = torch.empty(size * M, dtype=torch.uint8, device='cuda')
        del value
        graph.pop()
    escaped = []
    for iteration, size in enumerate((8, 16, 8, 16)):
        graph.push('large')
        value = torch.empty(size * M, dtype=torch.uint8, device='cuda')
        value.fill_(17 + iteration)
        escaped.append(value)
        del value
        graph.pop()
        graph.pop()  # Root pop finalizes handoffs; nested pop alone does not.
        for index, tensor in enumerate(escaped):
            assert tensor[0].item() == 17 + index
            assert tensor[-1].item() == 17 + index
        del tensor
        graph.push()
    if finish == 'abort':
        graph.abort()
    else:
        graph.pop()
    del graph
    gc.collect()
    for index, tensor in enumerate(escaped):
        assert tensor[0].item() == tensor[-1].item() == 17 + index
    del tensor, escaped
    gc.collect()
    torch.cuda.synchronize()
    assert aimdo.get_total_vram_usage() == baseline
aimdo.deinit()
print('Repeated escaped alias handoff, replay, abort and destruction passed')
