import numpy as np

def _unbroadcast_grad(parent_shape, grad):
    """
    Handles the gradient sum for a broadcasted operation.
    'parent_shape' is the shape of the tensor that was broadcasted.
    'grad' is the gradient from the child operation.
    """
    # 1. Handle new dims added by broadcasting (e.g., (3,) -> (4, 3))
    ndim_delta = grad.ndim - len(parent_shape)
    if ndim_delta > 0:
        grad = grad.sum(axis=tuple(range(ndim_delta))) # keepdims=False

    # 2. Handle dims that were 1 (e.g., (1, 3) -> (4, 3))
    axes_to_sum = tuple(i for i, dim in enumerate(parent_shape) if dim == 1 and grad.shape[i] > 1)
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)

    return grad

class Tensor:
    """
    A simple Tensor class to represent multi-dimensional arrays.
    This implementation is for Phase 1: Forward Pass Only.
    
    Attributes:
        data (np.ndarray): A Python list, scalar or np.ndarray.
        requires_grad (bool): Flag indicating if gradient computation is needed.
    """

    def __init__(self,data, requires_grad=False, _ctx=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def item(self):
        return self.data.item()

    # --- 1. Core Mathematical Operations ---
    def __add__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad

        out_data = self.data + other_data
        out_requires_grad = self.requires_grad or other_req_grad

        return Tensor(out_data, requires_grad=out_requires_grad)

    def __mul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad

        out_data = self.data * other_data
        out_requires_grad = self.requires_grad or other_req_grad

        return Tensor(out_data, requires_grad=out_requires_grad)
    
    def __sub__(self, other):
        return self - (other * -1)
    
    def __pow__(self, other):
        other_Data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad

        out_data = self.data ** other.data
        out_requires_grad = self.requires_grad or other_req_grad
        return Tensor(out_data, requires_grad=out_requires_grad)
    
    def __matmul__(self,other):
        other_data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad 

        out_data = self.data @ other.data
        out_requires_grad = self.requires_grad or other_req_grad

        return Tensor(out_data, requires_grad=out_requires_grad)
    
    # --- 2. Reflected (Right-hand-side) Operations ---

    def __radd__(self,other):
        return self + other
    
    def __rmul__(self,other):
        return self * other
    
    def __rsub__(self,other):
        return Tensor(other) - self
    
    def __rpow__(self,other):
        return Tensor(other) ** self
    
# --- 3. Unary Operations ---

    def __neg__(self):
        return self * -1
    

# --- 4. Reduction & Element-wise Ops ---
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        # Gradient flag is inherited
        return Tensor(out_data, requires_grad=self.requires_grad)

    def mean(self, axis=None, keepdims=False):
        # Implement mean in terms of sum. This simplifies the backward pass in Phase 2.
        out_sum = self.sum(axis=axis, keepdims=keepdims)
        
        # Calculate N (the number of elements being averaged)
        # We divide the total number of elements by the number of elements in the output
        n = np.prod(self.data.shape) / np.prod(out_sum.data.shape)
        
        out_data = out_sum.data * (1.0 / n)
        
        return Tensor(out_data, requires_grad=self.requires_grad)

    def relu(self):
        out_data = np.maximum(0, self.data)
        return Tensor(out_data, requires_grad=self.requires_grad)

    def exp(self):
        out_data = np.exp(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad)

    def log(self):
        out_data = np.log(self.data)
        return Tensor(out_data, requires_grad=self.requires_grad)

    # --- 5. Manipulation Ops ---

    def reshape(self, *shape):
        out_data = self.data.reshape(*shape)
        return Tensor(out_data, requires_grad=self.requires_grad)

    def transpose(self, *axes):
        # We need to handle the case where axes are not specified
        if not axes:
            axes = None
        out_data = self.data.transpose(*axes)
        return Tensor(out_data, requires_grad=self.requires_grad)