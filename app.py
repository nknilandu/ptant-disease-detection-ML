import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)

model = load_model("model.h5")
print("Model loaded. Check http://127.0.0.1:5000/")

labels = {
    0: "Healthy",
    1: "Powdery",
    2: "Rust"
}

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_probably_leaf(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    img = cv2.resize(img, (225, 225))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 30:
        return False


    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (225 * 225)
    if edge_density < 0.02: # Reject if it's too smooth/flat
        return False

    # 3. Green Color Check
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / (225 * 225)

    # 4. Saturation Check
    s_mean = hsv[:,:,1].mean()
    if s_mean > 200:
        return False

    return green_ratio > 0.12


def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    predictions = model.predict(x)[0]
    return predictions


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)

        if not is_probably_leaf(file_path):
            return "Invalid image: please upload a clear plant leaf image"

        predictions = getResult(file_path)

        confidence = float(np.max(predictions))
        predicted_index = int(np.argmax(predictions))

        if confidence < 0.80:
            return "Uncertain prediction: please upload a clearer leaf image"

        predicted_label = labels[predicted_index]
        return f"{predicted_label} ({confidence * 100:.2f}%)"

    return None


if __name__ == "__main__":
    app.run(debug=True)