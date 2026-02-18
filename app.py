from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model("anemia_detection_model.h5", compile=False)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    hb_val = None

    if request.method == "POST":
        palm_file = request.files.get("palm")
        nail_file = request.files.get("nail")
        age = float(request.form.get("age"))
        gender = request.form.get("gender")

        if palm_file and nail_file:
            # Process Palm
            palm_img = np.frombuffer(palm_file.read(), np.uint8)
            palm_img = cv2.imdecode(palm_img, cv2.IMREAD_COLOR)
            palm_img = cv2.resize(palm_img, (224,224)) / 255.0
            palm_input = np.expand_dims(palm_img, axis=0)

            # Process Nail
            nail_img = np.frombuffer(nail_file.read(), np.uint8)
            nail_img = cv2.imdecode(nail_img, cv2.IMREAD_COLOR)
            nail_img = cv2.resize(nail_img, (224,224)) / 255.0
            nail_input = np.expand_dims(nail_img, axis=0)

            # Meta
            meta_input = np.array([[age/100.0, 1 if gender=="Male" else 0]], dtype=np.float32)

            # Predict
            hb, anemia_prob = model.predict({
                "palm": palm_input,
                "nail": nail_input,
                "meta": meta_input
            })
            hb_val = hb[0][0]
            result = "Anemic" if anemia_prob[0][0] > 0.5 else "Non-Anemic"

    return render_template("index.html", result=result, hb_val=hb_val)

if __name__ == "__main__":
    app.run(debug=True)
