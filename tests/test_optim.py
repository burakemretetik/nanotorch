import numpy as np
import sys
import os
import pytest

# --- Test Setup ---
# Add project root to path so we can import 'nanotorch'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nanotorch.tensor import Tensor
from nanotorch.optim.sgd import SGD # Import our new class
# --- End Setup ---

def test_sgd_minimization():
    """
    Tests if the SGD optimizer can minimize a simple quadratic function:
    loss = (w - 3)^2
    
    The optimizer should find that the optimal 'w' is 3.
    """
    # 1. Setup
    # Start 'w' at a value far from the target (e.g., 10.0)
    w = Tensor([10.0], requires_grad=True)
    
    # Use a relatively high learning rate for a fast test
    # We pass the parameters to optimize as a list: [w]
    optimizer = SGD([w], lr=0.1)
    
    # 2. Training Loop
    print("\nTraining to find w=3:")
    for i in range(60): # 60 steps should be more than enough
        
        # A. Zero the gradients from the previous step
        optimizer.zero_grad()
        
        # B. Calculate the loss
        # The derivative of (w-3)^2 is 2*(w-3)
        loss = (w - 3) ** 2
        
        # C. Calculate gradients
        loss.backward()
        
        # D. Update the weight
        optimizer.step()
        
        print(f"  Step {i+1}: w = {w.data[0]:.4f}, loss = {loss.data.item():.4f}")

    # 3. Check the result
    # After 50 steps, 'w' should be very close to 3.0
    assert np.allclose(w.data, [3.0], atol=1e-5), \
           f"Optimization failed. Expected w=3.0, got w={w.data[0]}"