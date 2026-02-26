import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load model once
model = tf.keras.models.load_model("anemia_detection_model.h5", compile=False)

# Preprocessing helper
def preprocess_image(img):
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0)

# Prediction function
def predict(palm, nail, age, gender):
    # Convert Gradio PIL images to OpenCV format
    palm = cv2.cvtColor(np.array(palm), cv2.COLOR_RGB2BGR)
    nail = cv2.cvtColor(np.array(nail), cv2.COLOR_RGB2BGR)

    palm_input = preprocess_image(palm)
    nail_input = preprocess_image(nail)

    # Meta input (normalize age, encode gender)
    meta_input = np.array([[age/100.0, 1 if gender == "Male" else 0]], dtype=np.float32)

    # Run prediction
    hb, anemia_prob = model.predict({
        "palm": palm_input,
        "nail": nail_input,
        "meta": meta_input
    })

    hb_val = hb[0][0]
    result = "Anemic" if anemia_prob[0][0] > 0.5 else "Non-Anemic"

    return f"Hemoglobin: {hb_val:.2f} g/dL\nPrediction: {result}"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Palm Image"),
        gr.Image(type="pil", label="Nail Image"),
        gr.Number(label="Age"),
        gr.Radio(choices=["Male", "Female"], label="Gender")
    ],
    outputs="text",
    title="Non-Invasive Anemia Detection",
    description="Upload palm and nail images, enter age and gender to predict anemia risk."
)

iface.launch()