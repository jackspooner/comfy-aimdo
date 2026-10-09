"""Replacement failures must report an error and retain escaped tensor storage."""
import ctypes
import gc
import os
from pathlib import Path
import re
import subprocess
import sys
os.environ['PYTORCH_ALLOC_CONF'] = 'backend:cudaMallocAsync'
import torch
import comfy_aimdo.control as aimdo

if len(sys.argv) == 1:
    for op in ('p_cuMemCreate', 'p_cuMemMap'):
        subprocess.run([sys.executable, __file__, op], check=True)
    print('Escaped tensor integrity after allocation/remapping failures passed')
    raise SystemExit(0)

M = 1024 * 1024
assert aimdo.init('cuda')
assert aimdo.init_device(torch.cuda.current_device())
torch.empty(1, device='cuda')
# Read the repository-owned ABI rather than duplicating dispatch field offsets.
header = (Path(__file__).resolve().parents[1] / 'src/gpu_dispatch.h').read_text()
fields = re.findall(r'PFN_\w+ (p_\w+);', header.split('typedef struct AimdoCudaDispatch {')[1].split('} AimdoCudaDispatch;')[0])
class Dispatch(ctypes.Structure):
    _fields_ = [(name, ctypes.c_void_p) for name in fields]
dispatch = Dispatch.in_dll(aimdo.lib, 'g_cuda')

for operation, signature in (
    ('p_cuMemCreate', (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_ulonglong)),
    ('p_cuMemMap', (ctypes.c_ulonglong, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulonglong, ctypes.c_ulonglong)),
):
    if operation != sys.argv[1]:
        continue
    graph = aimdo.record(torch.cuda.current_stream())
    for size in (8, 16, 24):
        graph.push('large')
        value = torch.empty(size * M, dtype=torch.uint8, device='cuda')
        del value
        graph.pop()
    graph.push('large')
    first = torch.empty(8 * M, dtype=torch.uint8, device='cuda')
    first.fill_(17)
    graph.pop()
    graph.pop()
    graph.push()
    graph.push('large')
    second = torch.empty(16 * M, dtype=torch.uint8, device='cuda')
    second.fill_(23)
    graph.pop()
    original = getattr(dispatch, operation)
    calls = []
    def fail(*args):
        calls.append(True)
        return 2  # CUDA_ERROR_OUT_OF_MEMORY, including create_page's bounded retry.
    callback = ctypes.CFUNCTYPE(ctypes.c_int, *signature)(fail)
    setattr(dispatch, operation, ctypes.cast(callback, ctypes.c_void_p).value)
    try:
        try:
            graph.pop()
        except RuntimeError as error:
            assert 'aimdo memory compile error' in str(error)
        else:
            raise AssertionError('Injected replacement failure was not reported')
    finally:
        setattr(dispatch, operation, original)
    assert calls
    del graph
    gc.collect()
    assert first[0].item() == first[-1].item() == 17
    assert second[0].item() == second[-1].item() == 23
    del first, second
    gc.collect()
    torch.cuda.synchronize()
aimdo.deinit()
print('Escaped tensor integrity after allocation/remapping failures passed')
