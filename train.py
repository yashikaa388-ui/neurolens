"""Train a 4-class brain tumor classifier (EfficientNetB0, two-stage fine-tuning).
Expected layout:  dataset/Training/<class>/*.jpg   dataset/Testing/<class>/*.jpg
Run from the folder that contains this file.
"""
import os, glob, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, applications as A
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D, S, B = "dataset", (224, 224), 32
os.makedirs("model", exist_ok=True)
names = sorted(d for d in os.listdir(f"{D}/Training") if os.path.isdir(f"{D}/Training/{d}"))
json.dump(names, open("model/class_names.json", "w"))
print("Classes:", names)

def files(split):
    p, l = [], []
    for i, n in enumerate(names):
        for f in glob.glob(f"{D}/{split}/{n}/*"):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")): p.append(f); l.append(i)
    return np.array(p), np.array(l)

tp, tl = files("Training"); ep, el = files("Testing")
idx = np.random.RandomState(42).permutation(len(tp)); nv = int(.15 * len(tp))
vp, vl = tp[idx[:nv]], tl[idx[:nv]]; tp, tl = tp[idx[nv:]], tl[idx[nv:]]
print(f"train {len(tp)} | val {len(vp)} | test {len(ep)}")

def load(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, S); img.set_shape(S + (3,))
    return tf.cast(img, tf.float32), tf.one_hot(label, len(names))

def augment(x, y):
    x = tf.image.random_flip_left_right(x)
    x = tf.clip_by_value(tf.image.random_brightness(x, 20.), 0., 255.)
    return x, y

def make(p, l, train=False):
    d = tf.data.Dataset.from_tensor_slices((p, l))
    if train: d = d.shuffle(len(p), seed=42)
    d = d.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train: d = d.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return d.batch(B).prefetch(tf.data.AUTOTUNE)

tr, va, te = make(tp, tl, True), make(vp, vl), make(ep, el)
print("element_spec:", tr.element_spec)

_b = A.EfficientNetB0(include_top=False, weights="imagenet", input_shape=S + (3,))
base = models.Model(_b.input, _b.output, name="backbone")   # wrapper so app.py can find it by name
base.trainable = False
inp = layers.Input(S + (3,))                                  # raw pixels 0-255 (EfficientNet rescales internally)
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(.3)(x)
out = layers.Dense(len(names), activation="softmax")(x)
model = models.Model(inp, out)

cb = [tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss")]
model.compile(tf.keras.optimizers.Adam(1e-4), "categorical_crossentropy", ["accuracy"])
model.fit(tr, validation_data=va, epochs=15, callbacks=cb)    # stage 1: head only

base.trainable = True                                          # stage 2: last block only
for l in base.layers[:-20]: l.trainable = False
for l in base.layers[-20:]: l.trainable = not isinstance(l, layers.BatchNormalization)
model.compile(tf.keras.optimizers.Adam(1e-5), "categorical_crossentropy", ["accuracy"])
model.fit(tr, validation_data=va, epochs=15, callbacks=cb)

model.save("model/brain_tumor_model.keras")

probs = model.predict(te)
y_pred = probs.argmax(1)
rep = classification_report(el, y_pred, target_names=names, digits=4)
auc = roc_auc_score(np.eye(len(names))[el], probs, multi_class="ovr")
print(rep, "\nROC-AUC (macro OvR):", auc)
open("model/test_report.txt", "w").write(rep + f"\nROC-AUC: {auc:.4f}\n")
cm = confusion_matrix(el, y_pred)
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(len(names)), names, rotation=45)
plt.yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)): plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout(); plt.savefig("model/confusion_matrix.png", dpi=150)
