import tensorflow as tf
from tensorflow.keras import layers


class SelfAttention(layers.Layer):
    def __init__(self, feature_dim, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        
    def build(self, input_shape):
        self.query = layers.Dense(self.feature_dim)
        self.key = layers.Dense(self.feature_dim)
        self.value = layers.Dense(self.feature_dim)
        
    def call(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.feature_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        output = tf.matmul(attention_weights, value)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim})
        return config


class CrossAttention(layers.Layer):
    def __init__(self, feature_dim, **kwargs):
        super(CrossAttention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        
    def build(self, input_shape):
        self.query = layers.Dense(self.feature_dim)
        self.key = layers.Dense(self.feature_dim)
        self.value = layers.Dense(self.feature_dim)
        
    def call(self, inputs):
        x, context = inputs  #
        
        query = self.query(x)
        key = self.key(context)
        value = self.value(context)
        
        # Attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.feature_dim, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        output = tf.matmul(attention_weights, value)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({"feature_dim": self.feature_dim})
        return config
