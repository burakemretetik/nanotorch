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
    
    # --- Zero grad and backward functions ---
    
    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
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
        
        if grad is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad)
        
        for v in reversed(topo):
            
            parents, backward_fn = v._ctx
            
            grad_output = v.grad
            
            parent_grads = backward_fn(grad_output)
            
            if not isinstance(parent_grads, tuple):
                parent_grads = (parent_grads,)
                
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
                
                grad_self = _unbroadcast_grad(self.data.shape, grad_output)
                
                grad_other = None
                if isinstance(other, Tensor):
                    grad_other = _unbroadcast_grad(other.data.shape, grad_output)
                
                return tuple(g for g in (grad_self, grad_other) if g is not None)

            parents = tuple(t for t in (self, other) if isinstance(t, Tensor))
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
                grad_self = _unbroadcast_grad(self.data.shape, grad_output * other_data)
                grad_other = None
                if isinstance(other, Tensor):
                    grad_other = _unbroadcast_grad(other.data.shape, grad_output * self.data)
                return tuple(g for g in (grad_self, grad_other) if g is not None)
            parents = tuple(t for t in (self, other) if isinstance(t, Tensor))
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
                grad_self = grad_output * (other_data * (self.data ** (other_data-1)))
                return (grad_self,)
            parents = (self,)
            _ctx = (parents, _backward)
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    
    def __matmul__(self,other):
        if not isinstance(other, Tensor):
            raise ValueError("Matrix multiplication requires 'other' to be a Tensor.")
        
        out_data = self.data @ other.data
        out_requires_grad = self.requires_grad or other.requires_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                grad_self = _unbroadcast_grad(self.data.shape, grad_output @ other.data.T)
                grad_other = _unbroadcast_grad(other.data.shape, self.data.T @ grad_output)
                return(grad_self, grad_other)
            parents = (self, other)
            _ctx = (parents, _backward)
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)
    
    # --- 2. Reflected (Right-hand-side) Operations ---

    def __radd__(self,other):
        return self + other
    
    def __rmul__(self,other):
        return self * other
    
    def __rsub__(self,other):
        return Tensor(other) - self
    
    def __rpow__(self,other):
        raise NotImplementedError("Autograd for scalar ** Tensor is not supported")
    
# --- 3. Unary Operations ---

    def __neg__(self):
        return self * -1
    

# --- 4. Reduction & Element-wise Ops ---
    def sum(self, axis=None, keepdims=False):
        original_shape = self.data.shape
        _axis = axis

        out_data = self.data.sum(axis=_axis, keepdims=keepdims)
        out_requires_grad = self.requires_grad

        _ctx = None
        if out_requires_grad:
            def _backward(grad_output):
                if _axis is None:
                    grad_self = np.ones(original_shape) * grad_output
                
                else:
                    output_shape_kept = np.sum(np.ones(original_shape),axis=_axis,keepdims=True).shape
                    grad_output_reshaped = np.reshape(grad_output, output_shape_kept)
                    grad_self = np.ones(original_shape) * grad_output_reshaped
                return(grad_self,)
            
            parents = (self,)
            _ctx = (parents, _backward)
        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def mean(self, axis=None, keepdims=False):
        out_sum = self.sum(axis=axis, keepdims=keepdims)
        n = np.prod(self.data.shape) / np.prod(out_sum.data.shape)
        return out_sum * (1.0 / n)

    def relu(self):
        out_data = np.maximum(0, self.data)
        out_requires_grad = self.requires_grad

        _ctx = None
        if out_requires_grad:
            
            def _backward(grad_output):
                relu_mask = (self.data > 0)
                grad_self = grad_output * relu_mask
                
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
                grad_self = grad_output * (1.0 / self.data)
                
                return (grad_self,)

            parents = (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

# --- 5. Manipulation Ops ---

    def reshape(self, *shape):
        original_shape = self.data.shape
        
        out_data = self.data.reshape(*shape)
        out_requires_grad = self.requires_grad

        _ctx = None
        if out_requires_grad:
            
            def _backward(grad_output):
                grad_self = grad_output.reshape(original_shape)
                
                return (grad_self,)

            parents = (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)

    def transpose(self, *axes):
        if not axes:
            axes = None
            inverse_axes = None
        else:
            inverse_axes = np.argsort(axes)
            
        out_data = self.data.transpose(*axes if axes else ())
        out_requires_grad = self.requires_grad

        _ctx = None
        if out_requires_grad:
            
            def _backward(grad_output):
                grad_self = grad_output.transpose(*inverse_axes if inverse_axes else ())
                
                return (grad_self,)

            parents = (self,)
            _ctx = (parents, _backward)

        return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)