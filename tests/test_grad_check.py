import numpy as np
import sys
import os
import pytest

# Add the project root to the path so we can import 'nanotorch'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nanotorch.tensor import Tensor

def check_grad(tensor: Tensor, func, h=1e-5, tolerance=1e-5):
    """
    A helper function that compares the analytical gradient (from .backward())
    with the numerical gradient (finite differences).
    
    :param tensor: The Tensor whose gradient we are checking.
    :param func: A function that takes a Tensor and returns a
                 scalar Tensor (the "loss").
    :param h: The small step 'h' for the finite difference formula.
    :param tolerance: How close the gradients need to be to pass.
    """
    
    # --- 1. Calculate Analytical Gradient (Your .backward() code) ---
    tensor.zero_grad()
    loss = func(tensor)
    loss.backward()
    analytical_grad = tensor.grad.copy()
    
    # --- 2. Calculate Numerical Gradient (The "dumb" estimate) ---
    num_grad = np.zeros_like(tensor.data)
    
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        
        original_value = tensor.data[idx]
        
        # Calculate f(x + h)
        tensor.data[idx] = original_value + h
        # --- FIX 2 (Warning) ---
        # Use .item() to get a scalar, not a 0-d array
        fx_plus_h = func(Tensor(tensor.data)).item()
        
        # Calculate f(x - h)
        tensor.data[idx] = original_value - h
        # --- FIX 2 (Warning) ---
        # Use .item() to get a scalar
        fx_minus_h = func(Tensor(tensor.data)).item()
        
        # Apply the formula
        num_grad[idx] = (fx_plus_h - fx_minus_h) / (2 * h)
        
        # Restore the original value
        tensor.data[idx] = original_value
        
        it.iternext()

    # --- 3. Compare ---
    print(f"\nChecking Tensor with shape: {tensor.data.shape}")
    print(f"  Analytical Grad (from backward()):\n{analytical_grad}")
    print(f"  Numerical Grad (estimate):\n{num_grad}")
    
    assert np.allclose(analytical_grad, num_grad, atol=tolerance), "Gradients do not match!"

# --- The Actual Test Cases ---

def test_grad_check_mul():
    a = Tensor([3.0], requires_grad=True)
    func = lambda x: (x * x) # f(x) = x^2
    check_grad(a, func)

def test_grad_check_add():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    func = lambda x: (x + b).sum() # f(x) = sum(x + b)
    check_grad(a, func)

def test_grad_check_complex_chain():
    # f(x) = sum( ( (x * 2.0) + 1.0 ) ** 2 )
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    # --- FIX 1 (AttributeError) ---
    # Change .pow(2) to ** 2 and add .sum()
    func = lambda x: (((x * 2.0) + 1.0) ** 2).sum()
    check_grad(a, func)

def test_grad_check_relu():
    a = Tensor([-1.0, 0.5, 2.0], requires_grad=True)
    func = lambda x: x.relu().sum()
    check_grad(a, func)

def test_grad_check_matmul():
    a = Tensor(np.random.rand(4, 3), requires_grad=True)
    b = Tensor(np.random.rand(3, 2), requires_grad=True)
    func = lambda x: (x @ b).sum()
    check_grad(a, func)

def test_grad_check_broadcast_sum():
    # Tests _unbroadcast_grad for sum reduction
    a = Tensor(np.random.rand(4, 3), requires_grad=True)
    func = lambda x: x.sum()
    check_grad(a, func)

def test_grad_check_broadcast_op():
    # Tests _unbroadcast_grad for a binary operation
    a = Tensor(np.random.rand(1, 3), requires_grad=True)
    b = Tensor(np.random.rand(4, 1), requires_grad=True)
    func = lambda x: (x + b).sum() # a gets broadcasted to (4,3)
    check_grad(a, func)

