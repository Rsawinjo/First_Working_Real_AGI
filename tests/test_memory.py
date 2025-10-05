import pytest
from ai_core.memory_system import MemorySystem

def test_memory_persistence(tmp_path):
    mem = MemorySystem()
    test_data = {'foo': 'bar'}
    mem.save(test_data)
    loaded = mem.load()
    assert loaded == test_data

def test_memory_clear():
    mem = MemorySystem()
    mem.save({'foo': 'bar'})
    mem.clear()
    assert mem.load() == {}
