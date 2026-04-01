
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


    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
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



def build_no_attention(backbone="cnn1", mode="joint"):
    """Baseline — No attention, simple concatenation"""
    img_shape = (224, 224, 3)
    meta_dim  = 2

    palm_input = layers.Input(shape=img_shape, name="palm_input")
    nail_input = layers.Input(shape=img_shape, name="nail_input")
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")

    # ── Backbone ─────────────────────────────────────────────────
    palm_bb = get_backbone(backbone, img_shape, name="palm_backbone")
    nail_bb = get_backbone(backbone, img_shape, name="nail_backbone")

    palm_feat = palm_bb(palm_input)  # (7,7,128)
    nail_feat = nail_bb(nail_input)  # (7,7,128)

    # ── No attention — just pool and concatenate ──────────────────
    palm_pool = layers.GlobalAveragePooling2D()(palm_feat)  # (128,)
    nail_pool = layers.GlobalAveragePooling2D()(nail_feat)  # (128,)

    # ── MLP metadata ─────────────────────────────────────────────
    meta = layers.Dense(32,  activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(meta_input)
    meta = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(meta)

    # ── Simple concatenation ──────────────────────────────────────
    fused = layers.Concatenate()([palm_pool, nail_pool, meta])  # (384,)

    # ── Dense head ───────────────────────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)
    reg_out   = layers.Dense(1, activation="linear",  name="reg_output")(x)

    return Model(inputs=[palm_input, nail_input, meta_input],
                 outputs=[class_out, reg_out],
                 name="no_attention_joint")


def build_self_only(backbone="cnn1", mode="joint"):
    """Variant 1 — Self attention only"""
    img_shape = (224, 224, 3)
    meta_dim  = 2

    palm_input = layers.Input(shape=img_shape, name="palm_input")
    nail_input = layers.Input(shape=img_shape, name="nail_input")
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")

    palm_bb = get_backbone(backbone, img_shape, name="palm_backbone")
    nail_bb = get_backbone(backbone, img_shape, name="nail_backbone")

    palm_feat = palm_bb(palm_input)
    nail_feat = nail_bb(nail_input)

    # ── Self attention only ───────────────────────────────────────
    palm_self = self_attention(palm_feat, name="palm_self_attn")
    nail_self = self_attention(nail_feat, name="nail_self_attn")

    palm_pool = layers.GlobalAveragePooling1D()(palm_self)  # (128,)
    nail_pool = layers.GlobalAveragePooling1D()(nail_self)  # (128,)

    # ── MLP metadata ─────────────────────────────────────────────
    meta = layers.Dense(32,  activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(meta_input)
    meta = layers.Dense(128, activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(meta)

    # ── Concatenate ───────────────────────────────────────────────
    fused = layers.Concatenate()([palm_pool, nail_pool, meta])

    # ── Dense head ───────────────────────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)
    reg_out   = layers.Dense(1, activation="linear",  name="reg_output")(x)

    return Model(inputs=[palm_input, nail_input, meta_input],
                 outputs=[class_out, reg_out],
                 name="self_only_joint")


def build_self_cross(backbone="cnn1", mode="joint"):
    """Variant 2 — Self + Cross attention, no fusion attention"""
    img_shape = (224, 224, 3)
    meta_dim  = 2

    palm_input = layers.Input(shape=img_shape, name="palm_input")
    nail_input = layers.Input(shape=img_shape, name="nail_input")
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")

    palm_bb = get_backbone(backbone, img_shape, name="palm_backbone")
    nail_bb = get_backbone(backbone, img_shape, name="nail_backbone")

    palm_feat = palm_bb(palm_input)
    nail_feat = nail_bb(nail_input)

    # ── MLP metadata ─────────────────────────────────────────────
    meta = layers.Dense(32,  activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(meta_input)
    meta_out = layers.Dense(128, activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4))(meta)

    # ── Self + Cross attention ────────────────────────────────────
    palm_self  = self_attention(palm_feat,  name="palm_self_attn")
    nail_self  = self_attention(nail_feat,  name="nail_self_attn")
    palm_cross = cross_attention(palm_self, meta_out, name="palm_cross_attn")
    nail_cross = cross_attention(nail_self, meta_out, name="nail_cross_attn")

    palm_pool = layers.GlobalAveragePooling1D()(palm_cross)
    nail_pool = layers.GlobalAveragePooling1D()(nail_cross)

    # ── Concatenate (no fusion attention) ────────────────────────
    fused = layers.Concatenate()([palm_pool, nail_pool, meta_out])

    # ── Dense head ───────────────────────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)
    reg_out   = layers.Dense(1, activation="linear",  name="reg_output")(x)

    return Model(inputs=[palm_input, nail_input, meta_input],
                 outputs=[class_out, reg_out],
                 name="self_cross_joint")


def build_fusion_only(backbone="cnn1", mode="joint"):
    """Variant 3 — Fusion attention only, no self or cross"""
    img_shape = (224, 224, 3)
    meta_dim  = 2

    palm_input = layers.Input(shape=img_shape, name="palm_input")
    nail_input = layers.Input(shape=img_shape, name="nail_input")
    meta_input = layers.Input(shape=(meta_dim,), name="meta_input")

    palm_bb = get_backbone(backbone, img_shape, name="palm_backbone")
    nail_bb = get_backbone(backbone, img_shape, name="nail_backbone")

    palm_feat = palm_bb(palm_input)
    nail_feat = nail_bb(nail_input)

    # ── MLP metadata ─────────────────────────────────────────────
    meta = layers.Dense(32,  activation="relu",
                        kernel_regularizer=regularizers.l2(1e-4))(meta_input)
    meta_out = layers.Dense(128, activation="relu",
                            kernel_regularizer=regularizers.l2(1e-4))(meta)

    # ── No self or cross — go straight to fusion ─────────────────
    palm_pool = layers.GlobalAveragePooling2D()(palm_feat)  # (128,)
    nail_pool = layers.GlobalAveragePooling2D()(nail_feat)  # (128,)

    # ── Fusion attention only ─────────────────────────────────────
    fused = attention_fusion(
        layers.Reshape((1, 128))(palm_pool),
        layers.Reshape((1, 128))(nail_pool),
        meta_out
    )

    # ── Dense head ───────────────────────────────────────────────
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(fused)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64,  activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.2)(x)

    class_out = layers.Dense(1, activation="sigmoid", name="class_output")(x)
    reg_out   = layers.Dense(1, activation="linear",  name="reg_output")(x)

    return Model(inputs=[palm_input, nail_input, meta_input],
                 outputs=[class_out, reg_out],
                 name="fusion_only_joint")


