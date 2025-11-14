import numpy as np
from ..tensor import Tensor

class DataLoader:
    """
    A simple DataLoader to iterate over a dataset in mini-batches.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True):
        """
        Initializes the DataLoader.
        
        Args:
            dataset (tuple): A tuple of (X_data, y_data).
            batch_size (int): The number of samples per batch.
            shuffle (bool): If True, shuffle the data at the start of each epoch.
        """
        self.X, self.y = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = self.X.shape[0]

    def __iter__(self):
        """
        Called when a new iteration (epoch) starts.
        Handles shuffling and resetting the batch counter.
        """
        if self.shuffle:
            # Generate a shuffled permutation of indices
            indices = np.random.permutation(self.n_samples)
            self.X = self.X[indices]
            self.y = self.y[indices]
            
        self.current_batch = 0
        return self

    def __next__(self):
        """
        Called to get the next batch.
        """
        if self.current_batch * self.batch_size >= self.n_samples:
            # Reached the end of the dataset
            raise StopIteration
        
        # Calculate the start and end indices for this batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        # Get the batch data
        x_batch = self.X[start_idx:end_idx]
        y_batch = self.y[start_idx:end_idx]
        
        # Increment the batch counter
        self.current_batch += 1
        
        # Return the batch as Tensors
        # Note: We don't need gradients for the input data or targets
        return Tensor(x_batch), Tensor(y_batch)

    def __len__(self):
        """
        Returns the total number of batches in the dataset.
        """
        return (self.n_samples + self.batch_size - 1) // self.batch_size