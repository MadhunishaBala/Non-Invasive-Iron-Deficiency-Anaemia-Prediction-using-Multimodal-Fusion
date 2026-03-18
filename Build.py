
from tensorflow.keras import layers, regularizers, Model
from Backbones import get_backbone
from Attention import self_attention, cross_attention, attention_fusion

def build_models(backbone="cnn1", mode="classification",img_shape=(224, 224, 3), meta_dim=2):

    palm_input = layers.Input(shape=img_shape, name="palm_input")
    nail_input = layers.Input(shape=img_shape, name="nail_input")
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")
    
    palm_spatial = get_backbone(backbone, img_shape, "palm_backbone")(palm_input)
    nail_spatial = get_backbone(backbone, img_shape, "nail_backbone")(nail_input)

    meta_feat = layers.Dense(32,  activation="relu",kernel_regularizer=regularizers.l2(1e-4),name="meta_dense1")(meta_input)
    meta_feat = layers.Dense(128, activation="relu",kernel_regularizer=regularizers.l2(1e-4),name="meta_dense2")(meta_feat)

    palm_seq = self_attention(palm_spatial,  name="palm_self_attn")
    nail_seq = self_attention(nail_spatial,  name="nail_self_attn")
    palm_seq = cross_attention(palm_seq, meta_feat, name="palm_cross_attn")
    nail_seq = cross_attention(nail_seq, meta_feat, name="nail_cross_attn")

    fused = attention_fusion(palm_seq, nail_seq, meta_feat)


    x = layers.Dense(256, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation="relu",
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    
    if mode == "classification":
        out = layers.Dense(1, activation="sigmoid", name="output")(x)
    elif mode == "regression":
        out = layers.Dense(1, activation="linear",  name="output")(x)
    elif mode == "joint":
        class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)
        reg_out   = layers.Dense(1, activation="linear",  name="reg_output")(x)
        return Model(
            inputs  = [palm_input, nail_input, meta_input],
            outputs = [class_out, reg_out],
            name    = f"{backbone}_joint"
        )
    else:
        raise ValueError(f"mode must be 'classification', 'regression', or 'joint', got '{mode}'")

    return Model(
        inputs  = [palm_input, nail_input, meta_input],
        outputs = out,
        name    = f"{backbone}_{mode}"
    )