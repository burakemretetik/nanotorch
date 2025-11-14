import numpy as np
import sys
import os
import pytest

# --- Test Setup ---
# Add project root to path so we can import 'nanotorch'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nanotorch.tensor import Tensor
from nanotorch.data.dataloader import DataLoader
# --- End Setup ---

def test_dataloader_iteration():
    """
    Tests if the DataLoader correctly batches and iterates over the data.
    """
    # 1. Create a simple dummy dataset
    X_data = np.arange(100).reshape(10, 10) # 10 samples, 10 features
    y_data = np.arange(10) # 10 labels
    dataset = (X_data, y_data)
    
    batch_size = 4
    
    # 2. Initialize DataLoader
    # shuffle=False to make the test predictable
    loader = DataLoader(dataset, batch_size, shuffle=False)
    
    # 3. Check __len__
    # 10 samples / 4 batch_size = 2.5 -> 3 batches (4, 4, 2)
    assert len(loader) == 3
    
    # 4. Iterate over the loader
    batches = []
    for x_batch, y_batch in loader:
        batches.append((x_batch, y_batch))
        
    # 5. Check the batches
    assert len(batches) == 3
    
    # Check first batch
    assert isinstance(batches[0][0], Tensor) # Check type
    assert np.array_equal(batches[0][0].data, X_data[0:4])
    assert np.array_equal(batches[0][1].data, y_data[0:4])
    
    # Check second batch
    assert np.array_equal(batches[1][0].data, X_data[4:8])
    assert np.array_equal(batches[1][1].data, y_data[4:8])
    
    # Check third (and last) batch
    assert np.array_equal(batches[2][0].data, X_data[8:10])
    assert np.array_equal(batches[2][1].data, y_data[8:10])

def test_dataloader_shuffle():
    """
    Tests if the shuffle functionality works.
    """
    X_data = np.arange(100).reshape(10, 10)
    y_data = np.arange(10)
    dataset = (X_data, y_data)
    
    # No shuffle
    loader_no_shuffle = DataLoader(dataset, batch_size=10, shuffle=False)
    x_no_shuffle, _ = next(iter(loader_no_shuffle))
    
    # With shuffle
    loader_shuffle = DataLoader(dataset, batch_size=10, shuffle=True)
    x_shuffled, _ = next(iter(loader_shuffle))

    # Check that shuffling actually changed the order
    assert not np.array_equal(x_no_shuffle.data, x_shuffled.data)