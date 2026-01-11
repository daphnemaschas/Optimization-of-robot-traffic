
import torch
from src.buffer import Memory

def test_memory_storage():
    mem = Memory()
    # Simulate adding data
    mem.store(
        state=torch.randn(18), 
        global_state=torch.randn(54), 
        action=1, 
        reward=0.5, 
        is_terminal=False
    )
    # Assert one element
    assert len(mem.states) == 1
    # Test clearing the memory
    mem.clear()
    assert len(mem.states) == 0