import numpy as np
import sys
import os
import pytest

# --- Test Setup ---
# Add project root to path so we can import 'nanotorch'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nanotorch.tensor import Tensor
# Import the new modules
from nanotorch.nn.modules import Module, Linear, ReLU
# Import the functions
from nanotorch.functional import softmax, cross_entropy 

# Helper function for comparing arrays
def assert_allclose(a, b, atol=1e-6, msg=""):
    a_np = np.array(a) if not isinstance(a, np.ndarray) else a
    b_np = np.array(b) if not isinstance(b, np.ndarray) else b
    assert np.allclose(a_np, b_np, atol=atol), f"{msg} | Failed: {a_np} != {b_np}"
# --- End Setup ---


# --- 1. Test Module.parameters() ---

class SimpleNet(Module):
    """A test network to check nested parameter finding."""
    def __init__(self):
        super().__init__()
        # These should be found
        self.layer1 = Linear(10, 5)
        self.activation = ReLU() # This has no parameters, so it's fine
        self.layer2 = Linear(5, 1)
        # This should NOT be found (requires_grad=False)
        self.non_param = Tensor([1, 2, 3], requires_grad=False)
        # This should NOT be found (not a Tensor or Module)
        self.some_other_data = "hello"

def test_module_parameters():
    """Tests if the .parameters() method correctly finds all trainable Tensors."""
    net = SimpleNet()
    params = net.parameters()
    
    # We should find 4 parameters:
    # layer1.weight, layer1.bias, layer2.weight, layer2.bias
    assert len(params) == 4, "Did not find the correct number of parameters"
    
    # Check that they are all Tensors and require grad
    for p in params:
        assert isinstance(p, Tensor)
        assert p.requires_grad == True
    
    # Check shapes to be sure
    assert params[0].data.shape == (5, 10) # layer1.weight
    assert params[1].data.shape == (5,)   # layer1.bias
    assert params[2].data.shape == (1, 5)   # layer2.weight
    assert params[3].data.shape == (1,)   # layer2.bias

# --- 2. Test Linear layer forward/backward pass ---

def test_linear_forward_backward_zerograd():
    """
    Tests the full cycle: forward, backward, and zero_grad
    for the Linear layer.
    """
    # 1. Setup
    in_features, out_features = 10, 5
    batch_size = 4
    model = Linear(in_features, out_features)
    
    # Create random input
    x = Tensor(np.random.rand(batch_size, in_features), requires_grad=True)
    
    # 2. Forward pass
    y_pred = model.forward(x)
    
    # Check forward pass shape and grad status
    assert y_pred.data.shape == (batch_size, out_features)
    assert y_pred.requires_grad == True
    
    # 3. Backward pass
    # We use .sum() as a simple, arbitrary loss function.
    # This tests that the graph is connected from the output
    # all the way back to the model's parameters.
    loss = y_pred.sum()
    loss.backward()
    
    # 4. Check gradients
    params = model.parameters()
    assert len(params) == 2 # weight and bias
    
    weight_grad = params[0].grad
    bias_grad = params[1].grad
    
    assert weight_grad is not None
    assert bias_grad is not None
    
    assert weight_grad.shape == model.weight.data.shape
    assert bias_grad.shape == model.bias.data.shape
    
    # 5. Test zero_grad()
    model.zero_grad()
    assert params[0].grad is None, "Weight grad was not cleared"
    assert params[1].grad is None, "Bias grad was not cleared"

# --- 3. Test functional.py functions ---

def test_functional_softmax():
    """Tests the forward pass of softmax (it breaks the graph, as noted)."""
    x = Tensor([0, 1, 2], requires_grad=True)
    s = softmax(x)
    
    # Check if it computes correct values
    np_exps = np.exp([0, 1, 2])
    np_softmax = np_exps / np.sum(np_exps)
    
    assert_allclose(s.data, np_softmax)
    
    # Check that it sums to 1
    assert_allclose(s.sum().item(), 1.0)
    
    # We cannot test s.backward() because softmax() as written
    # breaks the graph (uses .data). This is expected.

def test_functional_cross_entropy():
    """
    Tests that our "smart" cross_entropy function
    correctly computes a forward pass AND a backward pass.
    """
    # 1. Setup
    batch_size, num_classes = 4, 10
    logits = Tensor(np.random.rand(batch_size, num_classes), requires_grad=True)
    targets = Tensor(np.random.randint(0, num_classes, size=batch_size))
    
    # 2. Forward pass
    loss = cross_entropy(logits, targets)
    
    assert isinstance(loss, Tensor)
    assert loss.requires_grad == True
    assert loss._ctx is not None, "cross_entropy did not create a graph"

    # 3. Backward pass
    loss.backward()
    
    # 4. Check gradient
    assert logits.grad is not None
    assert logits.grad.shape == logits.data.shape
    
    # The gradient (p - y) / N should not be all zeros
    assert not np.all(logits.grad == 0)