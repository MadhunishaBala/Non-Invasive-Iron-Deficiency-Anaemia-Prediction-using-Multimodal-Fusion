import tensorflow as tf

def self_attention_block(x, name="self_attention"):
    d = x.shape[-1]

    Q = tf.keras.layers.Dense(d, name=f"{name}_Q")(x)
    K = tf.keras.layers.Dense(d, name=f"{name}_K")(x)
    V = tf.keras.layers.Dense(d, name=f"{name}_V")(x)

    scores = tf.matmul(Q, K, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(d, tf.float32))

    weights = tf.nn.softmax(scores, axis=-1)

    output = tf.matmul(weights, V)
    return output


def cross_attention_block(query, key_value, name="cross_attention"):
    d = query.shape[-1]

    Q = tf.keras.layers.Dense(d, name=f"{name}_Q")(query)
    K = tf.keras.layers.Dense(d, name=f"{name}_K")(key_value)
    V = tf.keras.layers.Dense(d, name=f"{name}_V")(key_value)

    scores = tf.matmul(Q, K, transpose_b=True)
    scores = scores / tf.math.sqrt(tf.cast(d, tf.float32))

    weights = tf.nn.softmax(scores, axis=-1)

    output = tf.matmul(weights, V)
    return output
