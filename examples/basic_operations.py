try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    # If TensorFlow is not available, use our substitute
    from tensorflow_substitute import tf  # pylint: disable=import-error
    TF_AVAILABLE = False

import numpy as np
import time

# Basic TensorFlow operations
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)
print("Using real TensorFlow:", TF_AVAILABLE)

# TensorFlow constant with NumPy conversion
hello = tf.constant("Hello TensorFlow-NumPy!")
print(hello.numpy())

# Creating tensors and numpy arrays
tf_array = tf.constant([[1, 2, 3], [4, 5, 6]])
np_array = np.array([[1, 2, 3], [4, 5, 6]])

print("\nTensorFlow tensor:")
print(tf_array)
print("\nNumPy array:")
print(np_array)

# Convert between TensorFlow and NumPy
np_from_tf = tf_array.numpy()
tf_from_np = tf.constant(np_array)

print("\nNumPy array from TensorFlow:")
print(np_from_tf)
print("\nTensorFlow tensor from NumPy:")
print(tf_from_np)

# Basic math operations
print("\nMath operations:")
print("TensorFlow addition:", tf.add(tf_array, 10))
print("NumPy addition:", np_array + 10)

# Comparison
print("\nComparing execution:")
matrix_size = 500 if not TF_AVAILABLE else 1000  # Smaller matrices for substitute
tf_matrix = tf.random.uniform((matrix_size, matrix_size))
np_matrix = np.random.uniform(size=(matrix_size, matrix_size))

# Measure TensorFlow performance for matrix multiplication
start_time = time.time()
tf_result = tf.matmul(tf_matrix, tf.transpose(tf_matrix))
tf_time = time.time() - start_time

# Measure NumPy performance for matrix multiplication
start_time = time.time()
np_result = np.matmul(np_matrix, np_matrix.T)
np_time = time.time() - start_time

print(f"TensorFlow matrix multiplication time: {tf_time:.6f} seconds")
print(f"NumPy matrix multiplication time: {np_time:.6f} seconds")
if TF_AVAILABLE and tf_time < np_time:
    print(f"TensorFlow is {np_time/tf_time:.2f}x faster")
elif not TF_AVAILABLE:
    print(f"Performance ratio: {np_time/tf_time:.2f}x")
else:
    print(f"NumPy is {tf_time/np_time:.2f}x faster")