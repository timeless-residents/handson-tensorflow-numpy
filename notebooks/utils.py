import numpy as np

class TensorflowSubstitute:
    """A simple class to mimic basic TensorFlow functionality using NumPy.
    This allows examples to run without TensorFlow installed."""
    
    def __init__(self):
        self.__version__ = "substitute-1.0.0"
    
    class constant:
        @staticmethod
        def __call__(data):
            """Create a constant tensor-like object"""
            return TensorflowSubstitute.Tensor(data)
    
    class random:
        @staticmethod
        def uniform(shape):
            """Create a random uniform tensor-like object"""
            return TensorflowSubstitute.Tensor(np.random.uniform(size=shape))
    
    class Tensor:
        """A simple tensor-like wrapper around NumPy arrays"""
        def __init__(self, data):
            self.data = np.array(data)
        
        def numpy(self):
            """Return the underlying NumPy array"""
            return self.data
            
        def __str__(self):
            return str(self.data)
            
        def __repr__(self):
            return f"TensorflowSubstitute.Tensor({self.data})"
    
    @staticmethod
    def add(a, b):
        """Add two tensors or a tensor and a scalar"""
        if isinstance(a, TensorflowSubstitute.Tensor):
            a_data = a.data
        else:
            a_data = np.array(a)
            
        if isinstance(b, TensorflowSubstitute.Tensor):
            b_data = b.data
        else:
            b_data = np.array(b)
            
        return TensorflowSubstitute.Tensor(a_data + b_data)
    
    @staticmethod
    def matmul(a, b):
        """Matrix multiply two tensors"""
        if isinstance(a, TensorflowSubstitute.Tensor):
            a_data = a.data
        else:
            a_data = np.array(a)
            
        if isinstance(b, TensorflowSubstitute.Tensor):
            b_data = b.data
        else:
            b_data = np.array(b)
            
        return TensorflowSubstitute.Tensor(np.matmul(a_data, b_data))
    
    @staticmethod
    def transpose(a):
        """Transpose a tensor"""
        if isinstance(a, TensorflowSubstitute.Tensor):
            a_data = a.data
        else:
            a_data = np.array(a)
            
        return TensorflowSubstitute.Tensor(a_data.T)

# Create an instance that can be imported
tf = TensorflowSubstitute()
tf.constant = tf.constant()  # Make constant callable directly from tf