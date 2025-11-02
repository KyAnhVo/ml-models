import numpy as np
import numpy.typing as npt
from typing import List

from neuralnetwork import NeuralNetwork

class LogisticRegressor(NeuralNetwork):
    def __init__(self, attribute_count, lr):
        self.lr = np.float32(lr)
        self.attribute_count = attribute_count
        self.class_count = 1
        self.layer_count = 0
        
        self.weights = np.zeros(shape=(attribute_count + 1,), dtype=np.float32)
        self._output = np.zeros(shape=(1,), dtype=np.float32)
        self._error  = np.zeros(shape=(1,), dtype=np.float32)

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        x_biased = np.append(arr=x, values=1).astype(np.float32)
        self._attribute_inputs = x_biased.copy()
        self._output = self.sigmoid(self.weights @ x_biased)
        return np.array([self._output], dtype=np.float32)
        
    def backward(self, true_output: npt.NDArray[np.float32]):
        # Output error
        self._error = (self._output * (np.float32(1) - self._output) * (true_output - self._output)).astype(np.float32)

        # Weight calculation (look at NeuralNetwork for logic)
        e = self._error
        o = self._attribute_inputs
        grad = self.lr * o * e
        self.weights += grad

    def train_one(self, x: List[float], y: float):
        self.forward(np.array(x, dtype=np.float32))
        self.backward(np.array([y], dtype=np.float32))
        return np.array([self._output], dtype=np.float32)
        
def test_logistic_regressor():
    """Test the LogisticRegressor on simple binary classification problems"""
    print("\n" + "="*50)
    print("Testing LogisticRegressor (0 hidden layers)")
    print("="*50)
    
    # Test 1: AND gate (should work even with zero init)
    print("\n1. AND Gate Test:")
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    y = np.array([[0], [0], [0], [1]], dtype=np.float32)
    
    lr_model = LogisticRegressor(attribute_count=2, lr=0.5)
    
    # Train
    for epoch in range(1000):
        total_error = 0
        for i in range(len(X)):
            output = lr_model.forward(X[i])
            lr_model.backward(y[i])
            total_error += (output[0] - y[i][0]) ** 2
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch:4d}, MSE: {total_error/len(X):.6f}")
    
    # Test
    print("  Final predictions:")
    correct = 0
    for i in range(len(X)):
        pred = lr_model.forward(X[i])
        pred_class = 1 if pred[0] > 0.5 else 0
        actual = int(y[i][0])
        correct += (pred_class == actual)
        print(f"    {X[i]} → {pred[0]:.3f} (class: {pred_class}, target: {actual})")
    print(f"  Accuracy: {correct}/{len(X)} = {correct*100/len(X):.1f}%")
    
    # Test 2: OR gate
    print("\n2. OR Gate Test:")
    y_or = np.array([[0], [1], [1], [1]], dtype=np.float32)
    
    lr_or = LogisticRegressor(attribute_count=2, lr=0.5)
    
    # Quick train
    for epoch in range(500):
        for i in range(len(X)):
            output = lr_or.forward(X[i])
            lr_or.backward(y_or[i])
    
    # Quick test
    correct = 0
    for i in range(len(X)):
        pred = lr_or.forward(X[i])
        pred_class = 1 if pred[0] > 0.5 else 0
        actual = int(y_or[i][0])
        correct += (pred_class == actual)
    print(f"  Final Accuracy: {correct}/{len(X)} = {correct*100/len(X):.1f}%")
    
    # Test 3: Linearly separable 2D problem
    print("\n3. Linear Separation Test:")
    # Points above y=x are class 1, below are class 0
    X_linear = np.array([
        [0.2, 0.1], [0.3, 0.2], [0.1, 0.05],  # Below line (class 0)
        [0.2, 0.5], [0.4, 0.7], [0.6, 0.9]    # Above line (class 1)
    ], dtype=np.float32)
    y_linear = np.array([[0], [0], [0], [1], [1], [1]], dtype=np.float32)
    
    lr_linear = LogisticRegressor(attribute_count=2, lr=1.0)
    
    # Train
    for epoch in range(2000):
        for i in range(len(X_linear)):
            output = lr_linear.forward(X_linear[i])
            lr_linear.backward(y_linear[i])
    
    # Test
    correct = 0
    for i in range(len(X_linear)):
        pred = lr_linear.forward(X_linear[i])
        pred_class = 1 if pred[0] > 0.5 else 0
        actual = int(y_linear[i][0])
        correct += (pred_class == actual)
    print(f"  Final Accuracy: {correct}/{len(X_linear)} = {correct*100/len(X_linear):.1f}%")
    
    print("\n" + "="*50)
    print("LogisticRegressor tests complete!")
    return lr_model


if __name__ == "__main__":
    # Test the logistic regressor
    model = test_logistic_regressor()
    
    # Check weight updates are happening
    print(f"\nFinal weights (should be non-zero): {model.weights}")
