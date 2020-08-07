"""Calculates memory statistics about gpu usage."""
import torch


def stat_cuda(msg: str) -> None:
    """
    Calcualtes the current memory usage, the max memory usage of the program, the current memory
    cached and the max memory cached of the program.

    Args:
        msg (str): message to append before the information
    """
    print(f'-- {msg:<35} allocated: %dM, max allocated: %dM, cached: %dM, max cached: %dM' % (
        torch.cuda.memory_allocated() / 1024 / 1024,
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        torch.cuda.memory_cached() / 1024 / 1024,
        torch.cuda.max_memory_cached() / 1024 / 1024
    ))
