"""polite-run.py - run another Python script at below-normal priority with four CPU threads.
The machine is shared and an 80% load line binds; the model's own script knows nothing of that.
Usage: polite-run.py <script.py> [its args...]"""
import ctypes
import runpy
import sys

import torch
try:
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
except Exception:
    pass
torch.set_num_threads(4)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
