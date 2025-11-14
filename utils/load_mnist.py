import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist_data():
    """
    Fetches the MNIST dataset and splits it into train/test sets.
    
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    print("Fetching MNIST dataset... (This might take a moment)")
    
    # fetch_openml downloads the dataset. 
    # as_frame=False returns NumPy arrays
    try:
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    except Exception as e:
        print(f"Error fetching MNIST: {e}")
        print("Please check your internet connection or try again later.")
        return None, None
    
    print("Dataset fetched.")

    # --- Preprocessing ---
    # 1. Normalize pixel values from [0, 255] to [0, 1]
    X = X.astype(np.float32) / 255.0
    
    # 2. Convert labels from strings to integers
    y = y.astype(np.int64) # Use int64 for cross_entropy
    
    # 3. Split into training and test sets (60k train, 10k test)
    # The original MNIST split is 60k/10k. 
    # We use train_test_split to mimic this common practice.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, shuffle=True
    )
    
    print(f"X_train shape: {X_train.shape}") # (60000, 784)
    print(f"y_train shape: {y_train.shape}") # (60000,)
    print(f"X_test shape: {X_test.shape}")   # (10000, 784)
    print(f"y_test shape: {y_test.shape}")   # (10000,)
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    # Allows running this script directly to test the download
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    if X_train is not None:
        print("\nSuccessfully downloaded and processed MNIST.")