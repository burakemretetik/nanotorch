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
        grad = grad.sum(axis=tuple(range(ndim_delta)))

    # 2. Handle dims that were 1 (e.g., (1, 3) -> (4, 3))
    axes_to_sum = tuple(i for i, dim in enumerate(parent_shape) if dim == 1 and grad.shape[i] > 1)
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)

    return grad

class Tensor:
    """
    A simple Tensor class to represent multi-dimensional arrays.
    
    Attributes:
        data (np.ndarray): A Python list, scalar or np.ndarray.
        requires_grad (bool): Flag indicating if gradient computation is needed.
    """

    def __init__(self, data, requires_grad=False, _ctx=None):
        self.data = np.array(data)
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"
    
    def item(self):
        return self.data.item()
    
    # --- 0. Core Backward Operation
    def zero_grad(self):
        """Sets the gradient of this tensor to None."""
        self.grad = None

    def backward(self, grad=None):
        """Runs the backward pass (autograd engine) starting from this tensor."""
        
        # --- 1. Build Topological Sort ---
        topo = []
        visited = set()
        def _build_topo(v):
            if v not in visited:
                visited.add(v)
                if v._ctx:
                    for parent in v._ctx[0]:
                        _build_topo(parent)
                    topo.append(v)
        
        _build_topo(self)
        
        # --- 2. Initialize Starting Gradient ---
        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad)
            
        # --- 3. Propagate Gradients Backwards ---
        for v in reversed(topo):
            parents, backward_fn = v._ctx
            grad_output = v.grad
            parent_grads = backward_fn(grad_output)
            
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
                
            # --- 4. Accumulate Gradients in Parents ---
            for p, g in zip(parents, parent_grads):
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = g.copy() 
                    else:
                        p.grad += g

    # --- 1. Core Mathematical Operations ---
    def __add__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad

        out_data = self.data + other_data
        out_requires_grad = self.requires_grad or other_req_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output
                grad_other = grad_output
                
                # Handle broadcasting
                if self.requires_grad:
                    grad_self = _unbroadcast_grad(self.data.shape, grad_self)
                if isinstance(other, Tensor) and other.requires_grad:
                    grad_other = _unbroadcast_grad(other.data.shape, grad_other)
                
                # Return based on what needs gradients
                if isinstance(other, Tensor):
                    return (grad_self, grad_other)
                else:
                    return (grad_self,)
            
            parents = (self, other) if isinstance(other, Tensor) else (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def __mul__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad

        out_data = self.data * other_data
        out_requires_grad = self.requires_grad or other_req_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output * other_data
                grad_other = grad_output * self.data
                
                # Handle broadcasting
                if self.requires_grad:
                    grad_self = _unbroadcast_grad(self.data.shape, grad_self)
                if isinstance(other, Tensor) and other.requires_grad:
                    grad_other = _unbroadcast_grad(other.data.shape, grad_other)
                
                if isinstance(other, Tensor):
                    return (grad_self, grad_other)
                else:
                    return (grad_self,)
            
            parents = (self, other) if isinstance(other, Tensor) else (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)
    
    def __matmul__(self, other):
        """Matrix multiplication: self @ other"""
        other_data = other.data if isinstance(other, Tensor) else other
        other_req_grad = isinstance(other, Tensor) and other.requires_grad

        out_data = self.data @ other_data
        out_requires_grad = self.requires_grad or other_req_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                # For C = A @ B:
                # dL/dA = dL/dC @ B.T
                # dL/dB = A.T @ dL/dC
                grad_self = grad_output @ other_data.T
                grad_other = self.data.T @ grad_output
                
                if isinstance(other, Tensor):
                    return (grad_self, grad_other)
                else:
                    return (grad_self,)
            
            parents = (self, other) if isinstance(other, Tensor) else (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)
    
    def __sub__(self, other):
        return self + (other * -1)
    
    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError("Autograd for Tensor ** Tensor is not supported")

        other_data = other
        out_data = self.data ** other_data
        out_requires_grad = self.requires_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output * (other_data * (self.data ** (other_data - 1)))
                return (grad_self,) 

            parents = (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)
    
    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    # --- 2. Reflected Operations ---
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    def __rpow__(self, other):
        return Tensor(other) ** self
    
    def __neg__(self):
        return self * -1

    # --- 4. Reduction & Element-wise Ops ---
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out_requires_grad = self.requires_grad
        
        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                # Gradient of sum is broadcast back to original shape
                grad_self = grad_output
                
                # Expand dimensions that were reduced
                if axis is not None:
                    axes = (axis,) if isinstance(axis, int) else tuple(axis)
                    for ax in sorted(axes):
                        grad_self = np.expand_dims(grad_self, axis=ax)
                
                # Broadcast to original shape
                grad_self = np.broadcast_to(grad_self, self.data.shape)
                return (grad_self,)
            
            parents = (self,)
            _ctx = (parents, _backward)
        
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def mean(self, axis=None, keepdims=False):
        out_sum = self.sum(axis=axis, keepdims=keepdims)
        n = np.prod(self.data.shape) / np.prod(out_sum.data.shape)
        return out_sum / n

    def relu(self):
        out_data = np.maximum(0, self.data)
        out_requires_grad = self.requires_grad
        
        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output * (self.data > 0)
                return (grad_self,)
            
            parents = (self,)
            _ctx = (parents, _backward)
        
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def exp(self):
        out_data = np.exp(self.data)
        out_requires_grad = self.requires_grad
        
        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output * out_data
                return (grad_self,)
            
            parents = (self,)
            _ctx = (parents, _backward)
        
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def log(self):
        out_data = np.log(self.data)
        out_requires_grad = self.requires_grad
        
        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output / self.data
                return (grad_self,)
            
            parents = (self,)
            _ctx = (parents, _backward)
        
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    # --- 5. Manipulation Ops ---
    def reshape(self, *shape):
        out_data = self.data.reshape(*shape)
        out_requires_grad = self.requires_grad
        
        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output.reshape(self.data.shape)
                return (grad_self,)
            
            parents = (self,)
            _ctx = (parents, _backward)
        
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        elif len(axes) == 0:
            axes = None
        
        if not axes:
            inverse_axes = None
        else:
            inverse_axes = np.argsort(axes)
            
        out_data = self.data.transpose(axes)
        out_requires_grad = self.requires_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = grad_output.transpose(inverse_axes)
                return (grad_self,)

            parents = (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)