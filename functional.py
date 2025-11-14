import numpy as np
from .tensor import Tensor

def softmax(logits: Tensor, axis: int = -1) -> Tensor:
    """
    Computes stable softmax probabilities.
    
    NOTE: This function is "forward-only" because .max() is not
    implemented in our Tensor class. It breaks the graph.
    """
    # We use .data because .max() doesn't exist on our Tensor object
    logits_max = logits.data.max(axis=axis, keepdims=True)
    
    # 1. Subtract the max for numerical stability
    stable_logits = logits - logits_max # Uses broadcasting
    
    # 2. Exponentiate
    exps = stable_logits.exp()
    
    # 3. Normalize to get probabilities
    sum_exps = exps.sum(axis=axis, keepdims=True)
    
    return exps / sum_exps

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Computes the stable Cross-Entropy loss AND implements
    the backward pass as a "fused operation".
    
    Args:
        logits: Tensor of shape (batch_size, num_classes) - raw model outputs
        targets: Tensor of shape (batch_size,) - integer class indices (e.g., [0, 2, 1])
    """
    
    # --- 1. Forward Pass (using NumPy) ---
    # We compute the forward pass with NumPy because it requires indexing,
    # which our Tensor class does not support. The graph is created manually via _ctx.
    
    batch_size = logits.data.shape[0]
    num_classes = logits.data.shape[1]

    # A) Stable Logits (Log-Sum-Exp trick)
    logits_max = logits.data.max(axis=1, keepdims=True)
    stable_logits_data = logits.data - logits_max
    
    # B) Exponentiate and calculate sum-log
    exps_data = np.exp(stable_logits_data)
    sum_exps_data = exps_data.sum(axis=1, keepdims=True)
    log_sum_exp_data = np.log(sum_exps_data)
    
    # Final Log-Sum-Exp values
    log_sum_exp_final = (logits_max + log_sum_exp_data).reshape(batch_size,)
    
    # C) Get the logit for the correct target class (the part that requires indexing)
    targets_numpy = targets.data.astype(int)
    logits_for_targets = logits.data[np.arange(batch_size), targets_numpy]

    # D) Calculate log probabilities
    log_probs_data = logits_for_targets - log_sum_exp_final
    
    # E) Final Loss (Mean Negative Log Likelihood)
    out_data = np.mean(log_probs_data * -1.0)
    
    # --- 2. Create Graph (Context) ---
    _ctx = None
    out_requires_grad = logits.requires_grad
    
    if out_requires_grad:
        
        # This _backward function "remembers" values from the forward pass
        # (exps_data, sum_exps_data, targets_numpy, batch_size, num_classes)
        def _backward(grad_output):
            # This is the (p - y) "trick"
            
            # A) Calculate 'p' (the probabilities after softmax)
            p = exps_data / sum_exps_data 
            
            # B) Create 'y' (the one-hot labels)
            y = np.zeros((batch_size, num_classes))
            y[np.arange(batch_size), targets_numpy] = 1
            
            # C) Gradient is (p - y)
            #    We divide by batch_size because the forward pass used .mean()
            grad_logits = (p - y) / batch_size
            
            # D) Apply the upstream gradient (grad_output is a scalar, typically 1.0)
            grad_logits = grad_output * grad_logits
            
            # We only have one parent that needs a gradient: 'logits'
            return (grad_logits,)

        # The only TENSOR parent we track is 'logits'.
        # 'targets' are just data and don't require grad.
        parents = (logits,) 
        _ctx = (parents, _backward)
        
    # --- 3. Return the Loss Tensor ---
    return Tensor(out_data, requires_grad=out_requires_grad, _ctx=_ctx)