import numpy as np
from tensorflow_substitute import tf

# Basic operations
print("TensorFlow substitute version:", tf.__version__)
print("NumPy version:", np.__version__)

# TensorFlow constant with NumPy conversion
hello = tf.constant("Hello NumPy!")
print(hello.numpy())

# Creating tensors and numpy arrays
tf_array = tf.constant([[1, 2, 3], [4, 5, 6]])
np_array = np.array([[1, 2, 3], [4, 5, 6]])

print("\nTensorFlow substitute tensor:")
print(tf_array)
print("\nNumPy array:")
print(np_array)

# Convert between TensorFlow and NumPy
np_from_tf = tf_array.numpy()
tf_from_np = tf.constant(np_array)

print("\nNumPy array from TensorFlow substitute:")
print(np_from_tf)
print("\nTensorFlow substitute tensor from NumPy:")
print(tf_from_np)

# Basic math operations
print("\nMath operations:")
print("TensorFlow substitute addition:", tf.add(tf_array, 10))
print("NumPy addition:", np_array + 10)

# Comparison
print("\nComparing execution:")
tf_matrix = tf.random.uniform((500, 500))
np_matrix = np.random.uniform(size=(500, 500))

# Measure substitute performance for matrix multiplication
import time
start_time = time.time()
tf_result = tf.matmul(tf_matrix, tf.transpose(tf_matrix))
tf_time = time.time() - start_time

# Measure NumPy performance for matrix multiplication
start_time = time.time()
np_result = np.matmul(np_matrix, np_matrix.T)
np_time = time.time() - start_time

print(f"TensorFlow substitute matrix multiplication time: {tf_time:.6f} seconds")
print(f"NumPy matrix multiplication time: {np_time:.6f} seconds")
print(f"Performance ratio: {np_time/tf_time:.2f}x")