class SGD:
    """
    Implements Stochastic Gradient Descent (SGD) optimizer
    """

    def __init__(self, params: list, lr: float = 0.01):
        """
        Initializes the optimizer.
        
        Args:
            params (list): A list of Tensors (from model.parameters())
            lr (float): The learning rate.
        """
        self.params = params
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step (parameter update)
        It loops throgh all parameters and updates them using their gradients
        """
        for p in self.params:
            if p.requires_grad and p.grad is not None:
                # The core SGD update rule
                # p.data = p.data - (learning_rate * p.gradient)
                p.data -= self.lr * p.grad
    
    def zero_grad(self):
        """
        Clears the gradients of all parameters.

        Called before the .backward() pass 
        to prevent gradients from accumulating across epochs.
        """
        for p in self.params:
            p.zero_grad() # Calls Tensor.zero_grad()
    