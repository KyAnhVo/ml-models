import numpy as np
import numpy.typing as npt
from nn import NeuralNetwork
import matplotlib.pyplot as plt
from typing import Tuple, List


def test_xor() -> Tuple[float, List[float]]:
    """Test the network on XOR problem - the classic non-linear test"""
    print("\n" + "="*50)
    print("Testing XOR Gate (Non-linearly separable)")
    print("="*50)
    
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # Note: 2D for consistency
    
    nn = NeuralNetwork(
        attribute_count=2,
        layer_count=2,
        perceptron_per_layer_count=4,
        training_dat_count=4,
        lr=0.5,
        class_count=1
    )
    
    # Initialize with random weights (critical for XOR!)
    nn.enter_weights = (np.random.randn(*nn.enter_weights.shape) * 0.5).astype(np.float32)
    nn.inner_weights = (np.random.randn(*nn.inner_weights.shape) * 0.5).astype(np.float32)
    nn.exit_weights = (np.random.randn(*nn.exit_weights.shape) * 0.5).astype(np.float32)
    
    # Train
    errors = []
    for epoch in range(5000):
        epoch_error = 0.0
        for i in range(len(X)):
            output = nn.forward(X[i])
            nn.backward(y[i])
            epoch_error += np.sum((output - y[i]) ** 2)
        
        errors.append(epoch_error / len(X))
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d}, MSE: {errors[-1]:.6f}")
    
    # Test
    print("\nFinal predictions:")
    correct = 0
    for i in range(len(X)):
        pred = nn.forward(X[i])
        pred_class = 1 if pred[0] > 0.5 else 0
        actual_class = int(y[i][0])
        correct += (pred_class == actual_class)
        print(f"Input: {X[i]} Target: {actual_class} Prediction: {pred[0]:.3f} (class: {pred_class})")
    
    accuracy = correct / len(X)
    print(f"Accuracy: {correct}/{len(X)} = {accuracy*100:.1f}%")
    
    return accuracy, errors


def test_and_gate() -> float:
    """Test on AND gate - linearly separable, should work even with zero init"""
    print("\n" + "="*50)
    print("Testing AND Gate (Linearly separable)")
    print("="*50)
    
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
    y = np.array([[0], [0], [0], [1]], dtype=np.float32)
    
    nn = NeuralNetwork(
        attribute_count=2,
        layer_count=1,  # Single hidden layer should suffice
        perceptron_per_layer_count=2,
        training_dat_count=4,
        lr=0.3,
        class_count=1
    )
    
    # Test with zero initialization (should still work for AND)
    print("Training with zero initialization...")
    
    # Train
    for epoch in range(1000):
        for i in range(len(X)):
            output = nn.forward(X[i])
            nn.backward(y[i])
        
        if epoch % 200 == 0:
            error = sum((nn.forward(X[i]) - y[i])**2 for i in range(len(X))) / len(X)
            print(f"Epoch {epoch:4d}, MSE: {float(error[0]):.6f}")
    
    # Test
    print("\nFinal predictions:")
    correct = 0
    for i in range(len(X)):
        pred = nn.forward(X[i])
        pred_class = 1 if pred[0] > 0.5 else 0
        actual_class = int(y[i][0])
        correct += (pred_class == actual_class)
        print(f"Input: {X[i]} Target: {actual_class} Prediction: {pred[0]:.3f} (class: {pred_class})")
    
    accuracy = correct / len(X)
    print(f"Accuracy: {correct}/{len(X)} = {accuracy*100:.1f}%")
    
    return accuracy


def test_multiclass() -> float:
    """Test multi-class classification with 3 classes"""
    print("\n" + "="*50)
    print("Testing Multi-class Classification (3 classes)")
    print("="*50)
    
    # Simple 3-class problem: classify points in 2D space
    # Class 0: bottom-left, Class 1: top, Class 2: bottom-right
    X = np.array([
        [0, 0], [0.2, 0.1], [0.1, 0.2],  # Class 0
        [0.5, 0.9], [0.4, 0.8], [0.6, 0.7],  # Class 1
        [0.9, 0.1], [0.8, 0.2], [0.85, 0.15]  # Class 2
    ], dtype=np.float32)
    
    # One-hot encoded targets
    y = np.array([
        [1, 0, 0], [1, 0, 0], [1, 0, 0],  # Class 0
        [0, 1, 0], [0, 1, 0], [0, 1, 0],  # Class 1
        [0, 0, 1], [0, 0, 1], [0, 0, 1]   # Class 2
    ], dtype=np.float32)
    
    nn = NeuralNetwork(
        attribute_count=2,
        layer_count=2,
        perceptron_per_layer_count=5,
        training_dat_count=9,
        lr=0.3,
        class_count=3
    )
    
    # Initialize with random weights
    nn.enter_weights = (np.random.randn(*nn.enter_weights.shape) * 0.5).astype(np.float32)
    nn.inner_weights = (np.random.randn(*nn.inner_weights.shape) * 0.5).astype(np.float32)
    nn.exit_weights = (np.random.randn(*nn.exit_weights.shape) * 0.5).astype(np.float32)
    
    # Train
    print("Training...")
    for epoch in range(2000):
        for i in range(len(X)):
            output = nn.forward(X[i])
            nn.backward(y[i])
        
        if epoch % 400 == 0:
            error = sum(np.sum((nn.forward(X[i]) - y[i])**2) for i in range(len(X))) / len(X)
            print(f"Epoch {epoch:4d}, MSE: {error:.6f}")
    
    # Test
    print("\nFinal predictions:")
    correct = 0
    for i in range(len(X)):
        pred = nn.forward(X[i])
        pred_class = np.argmax(pred)
        actual_class = np.argmax(y[i])
        correct += (pred_class == actual_class)
        print(f"Input: {X[i]} Target: class {actual_class} "
              f"Prediction: [{pred[0]:.2f}, {pred[1]:.2f}, {pred[2]:.2f}] (class: {pred_class})")
    
    accuracy = correct / len(X)
    print(f"Accuracy: {correct}/{len(X)} = {accuracy*100:.1f}%")
    
    return accuracy


def test_weight_dimensions():
    """Test that weight dimensions are correct for various configurations"""
    print("\n" + "="*50)
    print("Testing Weight Dimensions")
    print("="*50)
    
    configs = [
        (2, 1, 3, 1),  # 2 inputs, 1 hidden layer, 3 nodes, 1 output
        (5, 2, 4, 3),  # 5 inputs, 2 hidden layers, 4 nodes, 3 outputs
        (10, 3, 8, 5), # 10 inputs, 3 hidden layers, 8 nodes, 5 outputs
    ]
    
    for attr, layers, nodes, classes in configs:
        nn = NeuralNetwork(
            attribute_count=attr,
            layer_count=layers,
            perceptron_per_layer_count=nodes,
            training_dat_count=10,
            lr=0.1,
            class_count=classes
        )
        
        print(f"\nConfig: {attr} inputs, {layers} layers, {nodes} nodes/layer, {classes} outputs")
        print(f"  enter_weights shape: {nn.enter_weights.shape} (expected: ({nodes}, {attr+1}))")
        print(f"  inner_weights shape: {nn.inner_weights.shape} (expected: ({layers-1}, {nodes}, {nodes+1}))")
        print(f"  exit_weights shape:  {nn.exit_weights.shape} (expected: ({classes}, {nodes+1}))")
        
        # Verify shapes
        assert nn.enter_weights.shape == (nodes, attr + 1), "Enter weights shape mismatch!"
        if layers > 1:
            assert nn.inner_weights.shape == (layers - 1, nodes, nodes + 1), "Inner weights shape mismatch!"
        assert nn.exit_weights.shape == (classes, nodes + 1), "Exit weights shape mismatch!"
        
        print("  ✓ All dimensions correct!")


def plot_learning_curves(errors_list: List[List[float]], labels: List[str]):
    """Plot learning curves for different tests"""
    plt.figure(figsize=(10, 6))
    
    for errors, label in zip(errors_list, labels):
        plt.plot(errors, label=label, alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curves')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("\nLearning curves saved to 'learning_curves.png'")


def main():
    print("\n" + "#"*60)
    print("#" + " "*58 + "#")
    print("#" + " "*15 + "NEURAL NETWORK TEST SUITE" + " "*18 + "#")
    print("#" + " "*58 + "#")
    print("#"*60)
    
    # Test weight dimensions first
    test_weight_dimensions()
    
    # Test AND gate (should work even with zero init)
    and_accuracy = test_and_gate()
    
    # Test XOR (needs random init)
    xor_accuracy, xor_errors = test_xor()
    
    # Test multi-class
    multiclass_accuracy = test_multiclass()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"AND Gate Accuracy:        {and_accuracy*100:.1f}%")
    print(f"XOR Gate Accuracy:        {xor_accuracy*100:.1f}%")
    print(f"Multi-class Accuracy:     {multiclass_accuracy*100:.1f}%")
    
    if xor_accuracy == 1.0:
        print("\n✅ SUCCESS: XOR problem solved perfectly!")
    elif xor_accuracy >= 0.75:
        print("\n⚠️  WARNING: XOR partially solved. May need more epochs or tuning.")
    else:
        print("\n❌ FAILURE: XOR not solved. Check backpropagation implementation.")
    
    # Optional: Plot learning curves
    try:
        plot_learning_curves([xor_errors], ["XOR Learning"])
    except ImportError:
        print("\nMatplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
