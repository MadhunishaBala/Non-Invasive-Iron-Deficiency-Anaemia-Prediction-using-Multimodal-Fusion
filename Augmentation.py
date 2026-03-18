import cv2
import numpy as np


def flip(img_uint8):
    return cv2.flip(img_uint8, 1)

def rotation(img_uint8, angle=15):
    h, w = img_uint8.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def brightness(img_uint8, factor=1.2):
    return np.clip(img_uint8.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def contrast(img_uint8, factor=1.2):
    mean = np.mean(img_uint8)
    return np.clip((img_uint8.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def augment_image(img):
    img_uint8 = (img * 255).astype(np.uint8)

    augmented = [
        flip(img_uint8),
        rotation(img_uint8, angle=15),
        brightness(img_uint8, factor=1.2),
        contrast(img_uint8, factor=1.2),
    ]

    return [a.astype(np.float32) / 255.0 for a in augmented]


def augment_training_set(X_palm_tr, X_nail_tr, X_meta_tr, y_lbl_tr):
    
    aug_palm, aug_nail, aug_meta, aug_label = [], [], [], []

    for i in range(len(y_lbl_tr)):
        # Keep original
        aug_palm.append(X_palm_tr[i][np.newaxis])
        aug_nail.append(X_nail_tr[i][np.newaxis])
        aug_meta.append(X_meta_tr[i][np.newaxis])
        aug_label.append(y_lbl_tr[i:i+1])
        
        palm_augs = augment_image(X_palm_tr[i])
        nail_augs = augment_image(X_nail_tr[i])

        for aug_idx in range(4):  # flip, rotation, brightness, contrast
            aug_palm.append(palm_augs[aug_idx][np.newaxis])
            aug_nail.append(nail_augs[aug_idx][np.newaxis])
            aug_meta.append(X_meta_tr[i][np.newaxis])   # same metadata
            aug_label.append(y_lbl_tr[i:i+1])           # same label

    total = len(y_lbl_tr) * 5
    perm  = np.random.permutation(total)

    return (
        np.concatenate(aug_palm,  axis=0)[perm],
        np.concatenate(aug_nail,  axis=0)[perm],
        np.concatenate(aug_meta,  axis=0)[perm],
        np.concatenate(aug_label, axis=0)[perm]
    )

def augment_training_set_joint(X_palm_tr, X_nail_tr, X_meta_tr, y_lbl_tr, y_hb_tr):
    
    aug_palm, aug_nail, aug_meta, aug_label, aug_hb = [], [], [], [], []

    for i in range(len(y_lbl_tr)):
        # Keep original
        aug_palm.append(X_palm_tr[i][np.newaxis])
        aug_nail.append(X_nail_tr[i][np.newaxis])
        aug_meta.append(X_meta_tr[i][np.newaxis])
        aug_label.append(y_lbl_tr[i:i+1])
        aug_hb.append(y_hb_tr[i:i+1])
        
        palm_augs = augment_image(X_palm_tr[i])
        nail_augs = augment_image(X_nail_tr[i])

        for aug_idx in range(4):
            aug_palm.append(palm_augs[aug_idx][np.newaxis])
            aug_nail.append(nail_augs[aug_idx][np.newaxis])
            aug_meta.append(X_meta_tr[i][np.newaxis])
            aug_label.append(y_lbl_tr[i:i+1])
            aug_hb.append(y_hb_tr[i:i+1])

    total = len(y_lbl_tr) * 5
    perm  = np.random.permutation(total)

    return (
        np.concatenate(aug_palm,  axis=0)[perm],
        np.concatenate(aug_nail,  axis=0)[perm],
        np.concatenate(aug_meta,  axis=0)[perm],
        np.concatenate(aug_label, axis=0)[perm],
        np.concatenate(aug_hb,    axis=0)[perm]
    )