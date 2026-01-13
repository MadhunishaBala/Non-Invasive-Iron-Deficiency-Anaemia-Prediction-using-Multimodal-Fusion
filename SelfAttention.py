import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class SelfAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.channels = input_shape[-1]
        
        # Create learnable weight matrices for Q, K, V
        self.query_conv = layers.Conv2D(
            filters=self.channels // 8,  # Reduce dimensionality
            kernel_size=1,
            name='query'
        )
        self.key_conv = layers.Conv2D(
            filters=self.channels // 8,
            kernel_size=1,
            name='key'
        )
        self.value_conv = layers.Conv2D(
            filters=self.channels,
            kernel_size=1,
            name='value'
        )
        
        # Learnable parameter to scale attention contribution
        self.gamma = self.add_weight(
            name='gamma',
            shape=[1],
            initializer='zeros',
            trainable=True
        )
        
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        batch_size, height, width, channels = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # Step 1: Generate Q, K, V from input
        query = self.query_conv(inputs)  # (batch, H, W, C//8)
        key = self.key_conv(inputs)      # (batch, H, W, C//8)
        value = self.value_conv(inputs)  # (batch, H, W, C)
        
        # Reshape to treat spatial dimensions as sequence
        # (batch, H*W, channels)
        query = tf.reshape(query, [batch_size, -1, tf.shape(query)[-1]])
        key = tf.reshape(key, [batch_size, -1, tf.shape(key)[-1]])
        value = tf.reshape(value, [batch_size, -1, channels])
        
        # Step 2 & 3: Calculate attention scores and scale
        # QK^T / sqrt(d_k)
        attention_scores = tf.matmul(query, key, transpose_b=True)  # (batch, H*W, H*W)
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        attention_scores = attention_scores / tf.sqrt(d_k)
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (batch, H*W, H*W)
        
        # Step 5: Apply attention weights to values
        # Output = Attention_Weights * V
        attended_values = tf.matmul(attention_weights, value)  # (batch, H*W, C)
        
        # Reshape back to spatial dimensions
        attended_values = tf.reshape(attended_values, [batch_size, height, width, channels])
        
        # Residual connection: weighted sum of input and attended output
        output = self.gamma * attended_values + inputs
        
        return output
    
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        return config


