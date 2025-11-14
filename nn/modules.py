import numpy as np
from ..tensor import Tensor

class Module:
    """
    Base class for all neural network modules.
    Layers (like linear) will inherit from this.
    """
    def parameters(self):
        """
        Returns a list of all trainable parameters (for the Tensors with requires_grad = True)
        """
        params = []
        # __dict__ holds all attributes of the class (self.weight, self.bias, ...)
        for name, value in self.__dict__.items():
            if isinstance(value, Tensor) and value.requires_grad:
                # If it's a trainable Tensor, add it
                params.append(value)
            elif isinstance(value, Module):
                # If it's a module, recursively ask it for its parameters
                params.extend(value.parameters())
        # FIX: Return statement was inside the loop!
        return params
    
    def zero_grad(self):
        """
        Calls .zero_grad() on all trainable parameters in the module.
        """
        for p in self.parameters():
            p.zero_grad()

class Linear(Module):
    """
    Implements a standard fully connected linear layer y = x @ W.T + b 
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Initialize parameters ---
        # He Initialization:
        he_std = np.sqrt(2.0 / in_features)
        w_data = np.random.randn(out_features, in_features) * he_std
        self.weight = Tensor(w_data, requires_grad=True)

        if self.use_bias:
            # Initialize biases to zero
            b_data = np.zeros(out_features)
            self.bias = Tensor(b_data, requires_grad=True)
        else:
            self.bias = None
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Performs the forward pass of the linear layer
        """
        # x @ W.T
        output = x @ self.weight.transpose(1, 0)
        if self.use_bias:
            # x @ W.T + b
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        # FIX: Was referencing self.bias instead of self.use_bias
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias})"

class ReLU(Module):
    """
    Implements the Rectified Linear Unit (ReLU) activation function as a module.
    """
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"