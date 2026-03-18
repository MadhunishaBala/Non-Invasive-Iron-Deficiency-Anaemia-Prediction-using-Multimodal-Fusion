from tensorflow.keras import layers, Model

def self_attention(x, num_heads=8, key_dim=64, name="self_attn"):
    h, w, c = x.shape[1], x.shape[2], x.shape[3]
    x_seq   = layers.Reshape((h * w, c))(x)
    attn    = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=name
    )(x_seq, x_seq)
    x_out   = layers.Add()([x_seq, attn])
    x_out   = layers.LayerNormalization()(x_out)
    return x_out


def cross_attention(query_seq, context, num_heads=8, key_dim=64, name="cross_attn"):
    dim          = query_seq.shape[-1]
    context_proj = layers.Dense(dim, activation="relu",
                                 name=f"{name}_proj")(context)
    context_seq  = layers.Reshape((1, dim))(context_proj)
    attn  = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=name
    )(query_seq, context_seq)
    x_out = layers.Add()([query_seq, attn])
    x_out = layers.LayerNormalization()(x_out)
    return x_out


def attention_fusion(palm_seq, nail_seq, meta_feat):
    palm_pooled = layers.GlobalAveragePooling1D()(palm_seq)
    nail_pooled = layers.GlobalAveragePooling1D()(nail_seq)
    dim         = palm_pooled.shape[-1]
    meta_proj   = layers.Dense(dim, activation="relu",
                                name="meta_proj_fusion")(meta_feat)

    palm_exp = layers.Reshape((1, dim))(palm_pooled)
    nail_exp = layers.Reshape((1, dim))(nail_pooled)
    meta_exp = layers.Reshape((1, dim))(meta_proj)
    stacked  = layers.Concatenate(axis=1)([palm_exp, nail_exp, meta_exp])

    fused    = layers.MultiHeadAttention(
        num_heads=4, key_dim=64, name="fusion_attn"
    )(stacked, stacked)
    fused    = layers.LayerNormalization()(fused + stacked)
    fused    = layers.Flatten()(fused)
    return fused