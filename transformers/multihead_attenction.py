
"""
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer 
sentence = "The quick brown fox jumps over the lazy dog"
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sentence)
input_tensor=tokenizer.texts_to_sequences(sentence)
print("input_tensor:",input_tensor)
input_tensor=tensorflow.convert_to_tensor(input_tensor)
print("input_tensor:",input_tensor)

"""
"""
import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention, Dense
sentence = "The quick brown fox jumps over the lazy dog"
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([sentence])
input_tensor = tokenizer.texts_to_sequences([sentence])[0]
print("input_tensor:",input_tensor)
input_tensor = tf.convert_to_tensor(input_tensor,dtype=float)
print("input_tensor:",input_tensor)
num_heads = 2
# Define the multi-head attention layer
multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=16)

# Apply the multi-head attention layer to the input tensor
attention_output = multi_head_attention(input_tensor, input_tensor)
dense_layer = Dense(32, activation='relu')

# Apply the dense layer to the output of the multi-head attention layer
dense_output = dense_layer(attention_output)

print("dense_output",dense_output)
"""
import tensorflow

import numpy as np

def scaled_dot_product_attention(Q, K, V, d_k):
  """Computes the scaled dot-product attention.

  Args:
    Q: The query matrix.
    K: The key matrix.
    V: The value matrix.
    d_k: The dimension of the keys.

  Returns:
    The attention matrix.
  """

  # Scale the dot product.
  scaled_dot_product = Q * K / np.sqrt(d_k)

  # Apply the softmax function.
  print("scaled_dot_product:",scaled_dot_product)
  attention = tensorflow.nn.softmax(scaled_dot_product, axis=-1)

  # Multiply the attention weights by the values.
  output = attention * V

  return output

def main():
  # Set the dimensions.
  d_model = 10
  d_k = 5

  # Create the query, key, and value matrices.
  Q = np.random.randn(1, 1, d_model)
  K = np.random.randn(1, 1, d_model)
  V = np.random.randn(1, 1, d_model)
  print("Q:",Q,"\n","K:",K,"\n","V:",V,"\n",d_k,d_model)

  # Compute the attention matrix.
  attention = scaled_dot_product_attention(Q, K, V, d_k)

  # Print the attention matrix.
  print(attention)

if __name__ == "__main__":
  main()
