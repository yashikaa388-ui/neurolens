import io, json, base64, datetime
import numpy as np, cv2, tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
model = tf.keras.models.load_model("model/brain_tumor_model.keras")
NAMES = json.load(open("model/class_names.json"))
base = model.get_layer("backbone")
head = model.layers[model.layers.index(base) + 1:]

INFO = {
 "glioma": ("Glioma", "Arises from glial cells. Can range from slow-growing to highly aggressive; grade needs biopsy and specialist assessment.", "Urgent neuro-oncology / neurosurgery referral."),
 "meningioma": ("Meningioma", "Arises from the meninges around the brain. Often benign and slow-growing, but location and size determine management.", "Neurosurgical review; may be monitored or treated."),
 "pituitary": ("Pituitary tumor", "Arises in the pituitary gland at the skull base. Most are benign but can affect hormones and vision.", "Endocrine tests and visual-field check, plus neurosurgical review."),
 "notumor": ("No tumor detected", "The model found no pattern matching the tumor classes it was trained on.", "A normal AI result does not rule out disease; follow the radiologist's read."),
}

def b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def gradcam(x, idx):
    with tf.GradientTape() as t:
        conv = base(x, training=False); t.watch(conv)
        y = conv
        for l in head: y = l(y)
        score = y[:, idx]
    g = t.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, -1))[0].numpy()
    return cam / (cam.max() + 1e-8)

@app.route("/")
def home(): return render_template("index.html")

@app.route("/health")
def health(): return "ok"

@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f: return jsonify(error="No image uploaded."), 400
    try: img = Image.open(f.stream).convert("RGB").resize((224, 224))
    except Exception: return jsonify(error="File is not a readable image (use JPG or PNG)."), 400
    arr = np.array(img); x = np.expand_dims(arr.astype("float32"), 0)
    p = model.predict(x, verbose=0)[0]; i = int(p.argmax()); key = NAMES[i]
    cam = cv2.resize(gradcam(tf.constant(x), i), (224, 224))
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.6 * arr + 0.4 * heat)
    title, desc, nxt = INFO.get(key, (key, "", ""))
    top2 = np.sort(p)[-2]
    return jsonify(
        positive=key != "notumor", label=key, title=title, description=desc, next_steps=nxt,
        confidence=float(p[i]), low_confidence=bool(p[i] < 0.75 or p[i] - top2 < 0.25),
        probs={n: float(v) for n, v in zip(NAMES, p)},
        original=b64(arr), gradcam=b64(overlay),
        time=datetime.datetime.now().strftime("%d %b %Y, %H:%M"))

if __name__ == "__main__":
    app.run(debug=True)
