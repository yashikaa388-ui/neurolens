import io, json, base64, html, datetime
import numpy as np, cv2, tensorflow as tf, streamlit as st
from PIL import Image

st.set_page_config(page_title="NeuroLens", page_icon="🧠", layout="wide")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root{--ink:#10222a;--mute:#64797f;--line:#dde7e9;--acc:#0b7a75;--acc2:#3fc1b9;--navy:#0b1b21}
.stApp{background:#f1f6f7;font-family:Inter,system-ui,sans-serif;color:var(--ink)}
#MainMenu,footer,header,[data-testid="stToolbar"]{visibility:hidden}
.block-container{padding:0 1rem 2rem;max-width:1180px}
@keyframes fade{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:1rem .3rem}
.logo{display:flex;align-items:center;gap:.6rem;font:700 1.25rem Fraunces,serif;color:var(--navy)}
.logo i{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,var(--acc),var(--acc2));display:inline-flex;align-items:center;justify-content:center;color:#fff;font-style:normal;font-size:1.1rem}
.tag{font-size:.82rem;color:var(--mute)}
.hero{background:radial-gradient(circle at 85% 20%,#12a39b 0,transparent 45%),linear-gradient(120deg,#08161b,#0b5f5b);color:#fff;border-radius:22px;padding:2.6rem 2.8rem;margin-bottom:1.4rem;animation:fade .6s ease}
.hero h1{font:700 2.6rem/1.1 Fraunces,serif;margin:0;color:#fff;letter-spacing:-.02em;max-width:720px;padding:0}
.hero p{margin:.9rem 0 0;color:#bfe1de;max-width:600px;font-size:1.05rem;line-height:1.6}
.stats{display:flex;gap:1.6rem;flex-wrap:wrap;margin-top:1.5rem}
.stats div{border-left:2px solid var(--acc2);padding-left:.8rem}
.stats b{display:block;font:700 1.5rem Fraunces,serif}.stats span{font-size:.8rem;color:#a9d3d0;text-transform:uppercase;letter-spacing:.08em}
[class*="st-key-card"]{background:#fff;border:1px solid var(--line);border-radius:18px;padding:1.3rem 1.4rem;box-shadow:0 4px 18px rgba(11,27,33,.06);margin-bottom:1rem;animation:fade .5s ease}
.h{font:600 1.15rem Fraunces,serif;color:var(--navy);margin:0 0 .6rem}
.sub{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;color:var(--acc);font-weight:600;margin-bottom:.2rem}
.step{display:flex;gap:.75rem;align-items:flex-start;margin:.6rem 0;color:#34494f;font-size:.93rem;line-height:1.45}
.step b{background:#e3f4f3;color:var(--acc);border-radius:50%;width:26px;height:26px;display:inline-flex;align-items:center;justify-content:center;font-size:.78rem;flex:none}
.verdict{border-radius:20px;padding:1.5rem 1.7rem;margin-bottom:1rem;color:#fff;animation:fade .5s ease;box-shadow:0 8px 24px rgba(0,0,0,.12)}
.verdict.pos{background:linear-gradient(120deg,#a61f19,#e5593f)}.verdict.neg{background:linear-gradient(120deg,#17653b,#35b26a)}
.verdict .l{font-size:.75rem;text-transform:uppercase;letter-spacing:.14em;opacity:.85}
.verdict .t{font:700 2rem/1.2 Fraunces,serif;margin:.2rem 0}.verdict .c{opacity:.95}
.meter{height:8px;background:rgba(255,255,255,.3);border-radius:99px;margin-top:.8rem;overflow:hidden}.meter i{display:block;height:100%;background:#fff;border-radius:99px}
.chips{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-bottom:1rem}
.chip{background:#fff;border:1px solid var(--line);border-radius:14px;padding:.8rem 1rem}
.chip small{color:var(--mute);display:block;font-size:.75rem;text-transform:uppercase;letter-spacing:.06em}.chip b{font-size:1.1rem}
.bar{margin:.8rem 0}.bar .n{display:flex;justify-content:space-between;font-size:.92rem;margin-bottom:.3rem}
.bar .tr{height:12px;background:#e6eef0;border-radius:99px;overflow:hidden}
.bar .fl{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--acc),var(--acc2))}
.bar.top .fl{background:linear-gradient(90deg,#c8392b,#ff8a65)}
.legend{height:10px;border-radius:99px;background:linear-gradient(90deg,#00008f,#00a0ff,#00ff80,#ffff00,#ff4000,#800000);margin-top:.5rem}
.legend+div{display:flex;justify-content:space-between;font-size:.75rem;color:var(--mute)}
.warn{background:#fff6e0;border:1px solid #f0c36d;border-radius:12px;padding:.8rem 1rem;color:#7a5a00;margin-bottom:1rem;font-size:.92rem}
.note{color:var(--mute);font-size:.84rem;line-height:1.5}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}
.empty{text-align:center;padding:3.2rem 1rem}.empty .big{font-size:3.2rem}
.foot{text-align:center;color:var(--mute);font-size:.82rem;padding:1.5rem 0 .5rem;border-top:1px solid var(--line);margin-top:1.5rem;line-height:1.7}
.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{font-weight:600;padding:.7rem 1.2rem;color:var(--mute)}
.stTabs [aria-selected="true"]{color:var(--acc)}
[data-testid="stFileUploader"] section{background:#f5fbfb;border:2px dashed var(--acc);border-radius:14px}
[data-testid="stFileUploader"] section:hover{background:#e9f7f6}
.stButton>button,.stDownloadButton>button{background:linear-gradient(120deg,var(--acc),#12a39b);color:#fff;border:0;border-radius:12px;padding:.7rem 1.4rem;font-weight:600;transition:.2s;box-shadow:0 4px 12px rgba(11,122,117,.3)}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px);color:#fff}
[data-testid="stImage"] img{border-radius:14px;border:1px solid var(--line)}
@media(max-width:700px){.hero{padding:1.6rem}.hero h1{font-size:1.8rem}.chips{grid-template-columns:1fr}}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

def test_acc():
    try:
        for ln in open("model/test_report.txt"):
            p = ln.split()
            if p and p[0] == "accuracy": return f"{float(p[-2])*100:.1f}%"
    except Exception:
        pass
    return None

INFO = {
 "glioma": ("Glioma", "Arises from glial cells. Can range from slow-growing to highly aggressive; grade needs biopsy and specialist assessment.", "Urgent neuro-oncology / neurosurgery referral."),
 "meningioma": ("Meningioma", "Arises from the meninges around the brain. Often benign and slow-growing, but location and size determine management.", "Neurosurgical review; may be monitored or treated."),
 "pituitary": ("Pituitary tumor", "Arises in the pituitary gland at the skull base. Most are benign but can affect hormones and vision.", "Endocrine tests and visual-field check, plus neurosurgical review."),
 "notumor": ("No tumor detected", "The model found no pattern matching the tumor classes it was trained on.", "A normal AI result does not rule out disease; follow the radiologist's read."),
}
LIMITS = ("This tool classifies a single 2D slice into four categories. It does not determine tumor grade (I-IV), size or exact "
          "location, and it cannot replace biopsy, full-volume MRI review or clinical judgement. It is a research prototype, not a medical device.")

@st.cache_resource(show_spinner="Loading AI model (first start only)…")
def load():
    m = tf.keras.models.load_model("model/brain_tumor_model.keras")
    names = json.load(open("model/class_names.json"))
    base = m.get_layer("backbone")
    head = m.layers[m.layers.index(base) + 1:]
    return m, names, base, head

def gradcam(base, head, x, idx):
    with tf.GradientTape() as t:
        conv = base(x, training=False); t.watch(conv)
        y = conv
        for l in head: y = l(y)
        score = y[:, idx]
    g = t.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, -1))[0].numpy()
    return cam / (cam.max() + 1e-8)

@st.cache_data(show_spinner="Analyzing scan…")
def analyze(data: bytes):
    model, names, base, head = load()
    arr = np.array(Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224)))
    x = np.expand_dims(arr.astype("float32"), 0)
    p = model.predict(x, verbose=0)[0]
    i = int(p.argmax()); key = names[i]
    cam = cv2.resize(gradcam(base, head, tf.constant(x), i), (224, 224), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    title, desc, nxt = INFO.get(key, (key, "", ""))
    top2 = np.sort(p)[-2]
    return dict(positive=key != "notumor", title=title, desc=desc, nxt=nxt, conf=float(p[i]),
                low=bool(p[i] < 0.75 or p[i] - top2 < 0.25),
                probs={n: float(v) for n, v in zip(names, p)}, orig=arr, cam=cam, heat=heat,
                time=datetime.datetime.now().strftime("%d %b %Y, %H:%M"))

def blend(r, s):
    a = ((r["cam"] ** 1.2) * s)[..., None]
    return np.uint8(r["orig"] * (1 - a) + r["heat"] * a)

def b64(arr):
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()

def report_html(r, pid, s):
    rows = "".join(f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>" for k, v in sorted(r["probs"].items(), key=lambda kv: -kv[1]))
    verdict = ("Tumor positive: " if r["positive"] else "Tumor negative: ") + r["title"]
    low = "<p style='color:#c8392b'><b>Low confidence: treat as unreliable and seek expert review.</b></p>" if r["low"] else ""
    return f"""<html><head><meta charset="utf-8"><title>NeuroLens report</title></head>
<body style="font-family:Arial;max-width:800px;margin:auto"><h1>MRI Analysis Report</h1>
<p>Case: {html.escape(pid or 'n/a')} | Generated: {r['time']}</p>
<h2>{verdict} ({r['conf']*100:.1f}% confidence)</h2>{low}
<table border="1" cellpadding="6" style="border-collapse:collapse"><tr><th>Class</th><th>Probability</th></tr>{rows}</table>
<h3>Visual explanation (Grad-CAM)</h3>
<img width="300" src="data:image/png;base64,{b64(r['orig'])}"> <img width="300" src="data:image/png;base64,{b64(blend(r, s))}">
<h3>About this finding</h3><p>{r['desc']}</p><h3>Suggested next step</h3><p>{r['nxt']}</p>
<h3>Limitations</h3><p>{LIMITS}</p></body></html>"""

acc = test_acc()
st.markdown('<div class="topbar"><div class="logo"><i>N</i>NeuroLens</div><div class="tag">Explainable AI for brain MRI</div></div>', unsafe_allow_html=True)
st.markdown(f"""<div class="hero"><h1>Brain tumor detection you can actually inspect.</h1>
<p>Upload an MRI slice. NeuroLens classifies it, shows exactly where the model looked, and prepares a report for clinical review.</p>
<div class="stats">{f'<div><b>{acc}</b><span>Test accuracy</span></div>' if acc else ''}<div><b>4</b><span>Tumor classes</span></div><div><b>Grad-CAM</b><span>Visual explanation</span></div></div></div>""", unsafe_allow_html=True)

left, right = st.columns([5, 7], gap="large")
with left:
    with st.container(key="card_upload"):
        st.markdown('<div class="sub">Step 1</div><div class="h">Upload your scan</div>', unsafe_allow_html=True)
        up = st.file_uploader("MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        pid = st.text_input("Case ID (optional)", placeholder="e.g. CASE-001")
    with st.container(key="card_how"):
        st.markdown("""<div class="h">How it works</div>
<div class="step"><b>1</b><span>The image is resized to 224×224 and normalized.</span></div>
<div class="step"><b>2</b><span>An EfficientNet model predicts glioma, meningioma, pituitary tumor or no tumor.</span></div>
<div class="step"><b>3</b><span>Grad-CAM highlights the regions that drove the prediction.</span></div>
<div class="step"><b>4</b><span>Download a report to print or save as PDF.</span></div>""", unsafe_allow_html=True)

with right:
    if up is None:
        with st.container(key="card_empty"):
            st.markdown('<div class="empty"><div class="big">🩻</div><div class="h">Results appear here</div><p class="note">Upload an axial brain MRI image to begin.<br>Images similar to the training data give the most reliable results.</p></div>', unsafe_allow_html=True)
    else:
        try:
            r = analyze(up.getvalue())
        except Exception:
            st.error("Could not read this file as an image."); st.stop()
        cls = "pos" if r["positive"] else "neg"
        st.markdown(f"""<div class="verdict {cls}"><div class="l">AI result</div>
<div class="t">{'Tumor positive' if r['positive'] else 'Tumor negative'} · {r['title']}</div>
<div class="c">{r['conf']*100:.1f}% confidence</div><div class="meter"><i style="width:{r['conf']*100:.1f}%"></i></div></div>""", unsafe_allow_html=True)
        if r["low"]:
            st.markdown('<div class="warn"><b>Low confidence.</b> The model is unsure. Treat this result as unreliable and seek expert review.</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="chips"><div class="chip"><small>Status</small><b>{'Positive' if r['positive'] else 'Negative'}</b></div>
<div class="chip"><small>Finding</small><b>{r['title']}</b></div><div class="chip"><small>Analyzed</small><b style="font-size:.9rem">{r['time']}</b></div></div>""", unsafe_allow_html=True)
        with st.container(key="card_result"):
            t1, t2, t3 = st.tabs(["Explanation", "Probabilities", "Report"])
            with t1:
                s = st.slider("Heatmap strength", 0, 100, 65) / 100
                c1, c2 = st.columns(2)
                c1.image(r["orig"], caption="Input scan", use_container_width=True)
                c2.image(blend(r, s), caption="Grad-CAM overlay", use_container_width=True)
                st.markdown('<div class="legend"></div><div><span>Low influence</span><span>High influence</span></div>', unsafe_allow_html=True)
                st.markdown('<p class="note" style="margin-top:.8rem">The heatmap shows where the model looked. It is not a tumor outline and can be wrong.</p>', unsafe_allow_html=True)
            with t2:
                items = sorted(r["probs"].items(), key=lambda kv: -kv[1])
                st.markdown("".join(f'<div class="bar {"top" if i == 0 and r["positive"] else ""}"><div class="n"><span>{INFO.get(k, (k,))[0]}</span><b>{v*100:.1f}%</b></div><div class="tr"><div class="fl" style="width:{v*100:.1f}%"></div></div></div>' for i, (k, v) in enumerate(items)), unsafe_allow_html=True)
            with t3:
                st.markdown(f'<div class="grid3"><div><div class="sub">Finding</div><p class="note">{r["desc"]}</p></div><div><div class="sub">Suggested next step</div><p class="note">{r["nxt"]}</p></div><div><div class="sub">Limitations</div><p class="note">{LIMITS}</p></div></div>', unsafe_allow_html=True)
                st.download_button("Download report (then Print > Save as PDF)", report_html(r, pid, 0.65),
                                   file_name="neurolens_report.html", mime="text/html")

st.markdown("""<div class="foot"><b>Research prototype, not a medical device.</b> Results must be reviewed by a qualified radiologist or clinician.<br>
Minor Project · School of Engineering and Technology, ITM University, Gwalior · Yashika Agarwal &amp; Ayushi Kumari · Guide: Mr Shubham Dhakarey</div>""", unsafe_allow_html=True)
