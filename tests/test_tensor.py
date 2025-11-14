import numpy as np
import sys
import os

# Add project root to path so we can import 'nanotorch'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from nanotorch.tensor import Tensor

# --- Helper function for all tests ---
def assert_allclose(a, b, atol=1e-6, msg=""):
    """ Helper to compare numpy arrays or scalars, with a message. """
    # Convert scalars to np.array for a consistent comparison
    a_np = np.array(a) if not isinstance(a, np.ndarray) else a
    b_np = np.array(b) if not isinstance(b, np.ndarray) else b
    
    assert np.allclose(a_np, b_np, atol=atol), f"{msg} | Failed: {a_np} != {b_np}"

# --- Phase 1: Forward Pass Tests ---

def test_initialization():
    """Tests tensor creation and default attributes."""
    a_data = [[1.0, 2.0], [3.0, 4.0]]
    a = Tensor(a_data, requires_grad=True)
    
    assert np.array_equal(a.data, np.array(a_data))
    assert a.requires_grad == True
    assert a.grad is None
    assert a._ctx is None

def test_binary_ops_forward_pass_and_grad_prop():
    """Tests forward pass for +, *, @ and requires_grad propagation."""
    # Test scalar op & grad prop (True * False -> True)
    a = Tensor([[1.0, 2.0]], requires_grad=True)
    b = a * 2.0
    assert np.array_equal(b.data, [[2.0, 4.0]])
    assert b.requires_grad == True

    # Test reflected scalar op
    b_reflected = 2.0 * a
    assert np.array_equal(b_reflected.data, b.data)
    assert b_reflected.requires_grad == True

    # Test tensor op & grad prop (True + False -> True)
    c = Tensor([[0.0, 1.0]], requires_grad=False)
    d = b + c
    assert np.array_equal(d.data, [[2.0, 5.0]])
    assert d.requires_grad == True
    
    # Test grad prop (False + False -> False)
    e = c + c
    assert e.requires_grad == False

    # Test matmul
    f = Tensor([[1., 2.]], requires_grad=True) # (1, 2)
    g = Tensor([[10.], [20.]], requires_grad=False) # (2, 1)
    h = f @ g
    assert h.data.shape == (1, 1)
    assert_allclose(h.item(), 50.0)
    assert h.requires_grad == True

def test_method_ops_forward_pass():
    """Tests forward pass for .sum(), .mean(), .relu(), .reshape()."""
    a = Tensor([[-1., 0.], [3., 4.]], requires_grad=True)
    
    # Test sum
    b = a.sum()
    assert_allclose(b.item(), 6.0)
    assert b.requires_grad == True
    
    # Test mean (full)
    c = a.mean()
    assert_allclose(c.item(), 1.5)
    assert c.requires_grad == True
    
    # Test mean (axis)
    d = a.mean(axis=0)
    assert_allclose(d.data, [1., 2.])

    # Test relu
    e = a.relu()
    assert_allclose(e.data, [[0., 0.], [3., 4.]])
    assert e.requires_grad == True
    
    # Test reshape
    f = a.reshape(4, 1)
    assert f.data.shape == (4, 1)
    assert f.requires_grad == True

# --- Phase 2: Graph Creation Tests ---

def test_graph_binary_op_tensors():
    """Tests _ctx is set correctly for Tensor + Tensor."""
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a + b
    assert c.requires_grad == True
    assert c._ctx is not None, "_ctx should not be None"
    assert c._ctx[0] == (a, b), "Parents are incorrect"
    assert callable(c._ctx[1]), "Backward function is not callable"

def test_graph_binary_op_scalar():
    """Tests _ctx is set correctly for Tensor * scalar."""
    d = Tensor([1, 2, 3], requires_grad=True)
    e = d * 2.0
    assert e.requires_grad == True
    assert e._ctx is not None, "_ctx should not be None"
    assert e._ctx[0] == (d,), "Parent should just be 'd'"
    assert callable(e._ctx[1])

def test_graph_no_grad():
    """Tests _ctx is None when requires_grad=False."""
    f = Tensor([1, 2, 3], requires_grad=False)
    g = Tensor([4, 5, 6], requires_grad=False)
    h = f + g
    assert h.requires_grad == False
    assert h._ctx is None, "_ctx should be None when no grad is required"

def test_graph_unary_op():
    """Tests _ctx for single-parent ops like relu."""
    i = Tensor([-1, 0, 1], requires_grad=True)
    j = i.relu()
    assert j.requires_grad == True
    assert j._ctx is not None
    assert j._ctx[0] == (i,)
    assert callable(j._ctx[1])

def test_graph_composite_op_mean():
    """Tests that .mean() builds a graph using .sum() and .*"""
    k = Tensor([1, 2, 3, 4], requires_grad=True)
    l = k.mean() # This should call .sum() and then .*
    assert l.requires_grad == True
    
    # 'l' is the result of 'mul', its parent is 'sum_tensor'
    assert l._ctx is not None, "mean() did not create a graph"
    sum_tensor = l._ctx[0][0] 
    assert_allclose(sum_tensor.item(), 10.0, msg="mean() parent is not sum")
    
    # The 'sum_tensor' parent is 'k'
    assert sum_tensor._ctx is not None, "sum() did not create a graph"
    assert sum_tensor._ctx[0][0] == k, "sum() parent is not k"

# --- Phase 2: Backward Pass (Autograd) Tests ---

def test_backward_simple_op():
    """Tests backward for c = (a*a).sum()"""
    a = Tensor([2.0], requires_grad=True)
    b = a * a
    c = b.sum()
    c.backward()
    assert_allclose(a.grad, [4.0])

def test_backward_chain_and_accumulation():
    """Tests d = (a*b) + a, checking gradient accumulation on 'a'."""
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)
    
    c = a * b  # c = 6.0
    d = c + a  # d = 8.0
    
    d.backward() 
    # d/da = (dc/da) + 1 = b + 1 = 3 + 1 = 4
    # d/db = (dc/db) = a = 2
    
    assert_allclose(a.grad, 4.0, msg="a.grad")
    assert_allclose(b.grad, 2.0, msg="b.grad")

def test_backward_sum_mean_and_zero_grad():
    """Tests .sum(), .zero_grad(), and .mean() backward passes."""
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    
    # Test sum
    b = a.sum() # b = 6.0
    b.backward()
    assert_allclose(a.grad, [1.0, 1.0, 1.0], msg="sum backward")
    
    # Test zero_grad
    a.zero_grad()
    assert a.grad is None
    
    # Test mean
    c = a.mean() # c = 2.0
    c.backward() # dc/da = [1/3, 1/3, 1/3]
    assert_allclose(a.grad, [1/3, 1/3, 1/3], msg="mean backward")

def test_backward_broadcasting():
    """Tests backward pass for a (1,3) + (2,1) broadcast."""
    a = Tensor([[1., 2., 3.]], requires_grad=True)       # shape (1, 3)
    b = Tensor([[4.], [5.]], requires_grad=True)       # shape (2, 1)
    c = a + b                                         # shape (2, 3)
    
    c.backward(np.ones_like(c.data)) # Start with grad of 1s
    
    # grad_a should be c.grad summed along axis 0
    assert_allclose(a.grad, [[2., 2., 2.]], msg="broadcast a.grad")
    # grad_b should be c.grad summed along axis 1
    assert_allclose(b.grad, [[3.], [3.]], msg="broadcast b.grad")

def test_backward_relu():
    a = Tensor([-2.0, 0.0, 3.0], requires_grad=True)
    b = a.relu() # b = [0.0, 0.0, 3.0]
    
    b.backward(np.array([1.0, 1.0, 1.0])) # Pass in upstream grad
    
    # Grad should be [0, 0, 1]
    assert_allclose(a.grad, [0.0, 0.0, 1.0])

def test_backward_matmul():
    a = Tensor([[1., 2.]], requires_grad=True) # (1, 2)
    b = Tensor([[10., 20.], [30., 40.]], requires_grad=True) # (2, 2)
    c = a @ b # c = [[70., 100.]] (1, 2)
    
    c.backward(np.array([[1., 1.]])) # Upstream grad
    
    # grad_a = grad_c @ b.T
    assert_allclose(a.grad, [[30., 70.]], msg="matmul a.grad")
    # grad_b = a.T @ grad_c
    assert_allclose(b.grad, [[1., 1.], [2., 2.]], msg="matmul b.grad")
