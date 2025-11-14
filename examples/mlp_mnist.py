import numpy as np
import sys
import os
import time

# --- 1. Add Project Root to Python Path ---
# This allows us to import our 'nanotorch' package
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- 2. Import All Our Bricks ---
from nanotorch.tensor import Tensor
from nanotorch.nn.modules import Module, Linear, ReLU
from nanotorch.optim.sgd import SGD
from nanotorch.functional import cross_entropy
from nanotorch.data.dataloader import DataLoader
from utils.load_mnist import load_mnist_data

# --- 3. Define the Neural Network ---
class SimpleMLP(Module):
    """A simple 2-layer Multi-Layer Perceptron (MLP) for MNIST."""
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # Input layer (784 features) -> Hidden layer (e.g., 128 features)
        self.layer1 = Linear(input_size, hidden_size)
        # Activation function
        self.relu = ReLU()
        # Hidden layer (128 features) -> Output layer (10 classes)
        self.layer2 = Linear(hidden_size, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        """The forward pass of the network."""
        x = self.layer1.forward(x)
        x = self.relu.forward(x)
        x = self.layer2.forward(x)
        # We return the raw "logits" (pre-softmax scores)
        return x

# --- 4. Main Training Script ---
if __name__ == "__main__":
    
    # --- Hyperparameters ---
    INPUT_SIZE = 784  # 28x28 pixels
    HIDDEN_SIZE = 128
    NUM_CLASSES = 10
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 5
    BATCH_SIZE = 64
    
    # --- Load Data ---
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    if X_train is None:
        sys.exit("Failed to load MNIST data.")
        
    # Create DataLoaders
    train_loader = DataLoader((X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    
    # --- Initialize Model, Loss, and Optimizer ---
    model = SimpleMLP(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES)
    # The loss function is just a function
    loss_fn = cross_entropy
    # The optimizer holds the model's parameters and the learning rate
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Training ---")
    
    start_time = time.time()
    loss_history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        # Iterate over batches
        for i, (x_batch, y_batch) in enumerate(train_loader):
            
            # 1. Zero gradients (from the previous step)
            optimizer.zero_grad()
            
            # 2. Forward Pass: Get model's predictions
            y_pred_logits = model.forward(x_batch)
            
            # 3. Calculate Loss
            loss = loss_fn(y_pred_logits, y_batch)
            
            # 4. Backward Pass: Compute gradients
            loss.backward()
            
            # 5. Optimizer Step: Update weights
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print loss at the end of the epoch
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    # --- 5. Evaluate the Model (Check Accuracy) ---
    # (Note: We use .data for evaluation since we don't need the graph)
    
    # Get predictions on the test set
    test_logits = model.forward(Tensor(X_test)).data
    
    # Get the index of the highest score (this is the predicted class)
    predicted_classes = np.argmax(test_logits, axis=1)
    
    # Compare with true labels
    correct_predictions = (predicted_classes == y_test).sum()
    accuracy = correct_predictions / y_test.shape[0]
    
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # --- 6. Plot and Save Loss Curve ---
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(1, NUM_EPOCHS + 1), loss_history)
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.savefig("mnist_loss_curve.png")
        print("Loss curve saved to 'mnist_loss_curve.png'")
    except ImportError:
        print("\n'matplotlib' not found. Skipping plot generation.")
        print("Install it with 'pip install matplotlib' to see the loss curve.")