import streamlit as st
# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Deteksi Penyakit Jeruk",
    page_icon="🍊",
    layout="wide"
)
import numpy as np
import tensorflow as tf
from PIL import Image
# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
/* ===== BACKGROUND ===== */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1611080626919-7cf5a9dbab5b?q=80&w=1974&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
/* ===== OVERLAY ===== */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.20);
    z-index: -1;
}
/* ===== HIDE STREAMLIT ===== */
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
header {
    visibility: hidden;
}
/* ===== REMOVE SIDEBAR ===== */
section[data-testid="stSidebar"] {
    display: none;
}
/* ===== GLOBAL TEXT ===== */
html, body, [class*="css"] {
    color: #f8fafc;
    font-family: 'Segoe UI', sans-serif;
}
/* ===== NAVBAR ===== */
div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    align-items: center;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(18px);
    padding: 18px 0;
    z-index: 9999;
    gap: 60px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 5px 20px rgba(0,0,0,0.45);
}
/* ===== NAV ITEM ===== */
div[role="radiogroup"] label {
    font-size: 18px;
    font-weight: 600;
    color: #f8fafc !important;
    transition: 0.3s ease;
    padding: 8px 18px;
    border-radius: 10px;
}
/* ===== HOVER ===== */
div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.08);
    transform: translateY(-2px);
    color: #facc15 !important;
}
/* ===== TITLE ===== */
.title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #facc15;
    margin-top: 120px;
    animation: fadeIn 1s ease-in-out;
    text-shadow: 0 4px 20px rgba(0,0,0,0.65);
}
.subtitle {
    text-align: center;
    color: #ffffff;
    font-size: 22px;
    margin-bottom: 40px;
    width: 85%;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.8;
    text-shadow: 0 2px 8px rgba(0,0,0,0.35);
}
/* ===== GLASS CARD ===== */
.card {
    background: rgba(15,23,42,0.72);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 10px 35px rgba(0,0,0,0.45);
    transition: all 0.3s ease;
    color: #f8fafc;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(251,146,60,0.25);
}
/* ===== ABOUT CARD ===== */
.about-card {
    background: rgba(15,23,42,0.72);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 28px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.06);
    transition: all 0.3s ease;
    color: #f8fafc;
    box-shadow: 0 10px 35px rgba(0,0,0,0.4);
}
.about-card:hover {
    transform: scale(1.01);
    box-shadow: 0 10px 35px rgba(251,146,60,0.25);
}
/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #f97316, #fb923c);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
    transition: 0.3s;
    font-weight: 600;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ea580c, #f97316);
}
/* ===== INPUT ===== */
.stSelectbox div[data-baseweb="select"],
.stSlider,
.stFileUploader,
.stCameraInput {
    background: rgba(15,23,42,0.65);
    border-radius: 14px;
    padding: 5px;
}
/* ===== METRIC ===== */
[data-testid="metric-container"] {
    background: rgba(15,23,42,0.72);
    border-radius: 20px;
    padding: 15px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.05);
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}
/* ===== PROGRESS ===== */
.stProgress > div > div {
    background: linear-gradient(to right, orange, yellow);
}
/* ===== RESULT CARD ===== */
.result-card {
    background: rgba(15,23,42,0.78);
    backdrop-filter: blur(22px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 30px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.45);
    transition: 0.3s ease;
    color: #f8fafc;
}
.result-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 45px rgba(251,146,60,0.25);
}
/* ===== RESULT TITLE ===== */
.result-title {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 25px;
    color: #ffffff;
}
/* ===== BADGE ===== */
.badge {
    display: inline-block;
    padding: 10px 20px;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.badge-green {
    background: linear-gradient(90deg,#22c55e,#16a34a);
}
.badge-yellow {
    background: linear-gradient(90deg,#eab308,#facc15);
    color: black;
}
.badge-red {
    background: linear-gradient(90deg,#ef4444,#dc2626);
}
/* ===== CONFIDENCE ===== */
.conf-box {
    margin-top: 18px;
    padding: 16px;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    font-size: 18px;
    backdrop-filter: blur(10px);
    color: #f8fafc;
    border: 1px solid rgba(255,255,255,0.05);
}
/* ===== PROGRESS LABEL ===== */
.prob-label {
    margin-top: 15px;
    margin-bottom: 5px;
    font-size: 16px;
    font-weight: 600;
    color: #f8fafc;
}
/* ===== IMAGE ===== */
img {
    border-radius: 18px;
}
/* ===== TEXT ===== */
p, li, h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    line-height: 1.7;
}
/* ===== ANIMATION ===== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
""", unsafe_allow_html=True)
# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_my_model():
    model = tf.saved_model.load("exported_model")
    return model.signatures["serving_default"]
model = load_my_model()
# =========================
# CLASS
# =========================
class_names = ['citrus_canker', 'healthy', 'melanose']
# =========================
# NAVBAR
# =========================
menu = st.radio(
    "",
    ["Home", "Deteksi", "Panduan", "About", "Kontak"],
    horizontal=True
)
# =========================
# FUNCTIONS
# =========================
def preprocess(image):
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
def confidence_label(conf):
    if conf > 0.9:
        return "Tinggi"
    elif conf > 0.7:
        return "Sedang"
    else:
        return "Rendah"
# =========================
# HOME
# =========================
if menu == "Home":

    st.markdown("""
    <div class="title">
         Deteksi Penyakit Buah Jeruk
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="subtitle">
    Sistem ini merupakan aplikasi berbasis Artificial Intelligence (AI)
    yang digunakan untuk mengidentifikasi buah jeruk dari citra digital.
    Teknologi yang digunakan dalam sistem ini adalah Deep Learning dengan
    metode Convolutional Neural Networks (CNN) dan MobileNetV2
    sebagai metode klasifikasi yang mampu mengenali pola visual pada citra digital.
    Sistem mampu mendeteksi 3 kondisi buah jeruk yaitu
    Healthy, Melanose, dan Citrus Canker.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # =========================
    # 3 KOLOM
    # =========================
    col1, col2, col3 = st.columns(3)

    # =========================
    # HEALTHY
    # =========================
    with col1:

        st.markdown("""
        <div class="card">
            <h2 style="text-align:center; color:#22c55e;">
                🟢 Healthy
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            "images/h.png",
            width=300
        )

        st.markdown("""
        <div class="card">
            Buah jeruk sehat tanpa indikasi penyakit.
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # MELANOSE
    # =========================
    with col2:

        st.markdown("""
        <div class="card">
            <h2 style="text-align:center; color:#facc15;">
                🟡 Melanose
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            "images/m.jpg",
            width=300
        )

        st.markdown("""
        <div class="card">
            Penyakit jamur yang menyebabkan bercak hitam kecil pada kulit buah.
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # CANKER
    # =========================
    with col3:

        st.markdown("""
        <div class="card">
            <h2 style="text-align:center; color:#ef4444;">
                🔴 Citrus Canker
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.image(
            "images/c.jpg",
            width=300
        )

        st.markdown("""
        <div class="card">
            Penyakit bakteri yang menyebabkan bercak coklat pada buah jeruk.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align:center;
        font-size:24px;
        font-weight:600;
        color:white;
        margin-top:20px;
        text-shadow:0 2px 10px rgba(0,0,0,0.5);
    ">
         Gunakan menu <b>Deteksi</b> untuk mulai mendeteksi gambar buah jeruk,
            Silakan gunakan menu about jika igin mengetahui sistem ini,
            jika bingung silakan ke menu panduan
    </div>
    """, unsafe_allow_html=True)
# =========================
# DETEKSI
# =========================
elif menu == "Deteksi":
    st.markdown("""
    <div class="title">
         Deteksi Penyakit Buah Jeruk
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
        Upload gambar atau gunakan kamera
    </div>
    """, unsafe_allow_html=True)
    threshold = st.slider(
        " Threshold Keyakinan",
        0.5,
        0.95,
        0.7
    )
    st.markdown("""
    <div class="card">
        <h4>📷 Sumber gambar</h4>
    </div>
    """, unsafe_allow_html=True)
    input_mode = st.selectbox(
        "",
        ["Upload", "Kamera"]
    )
    images = []
    if input_mode == "Upload":
        uploaded_files = st.file_uploader(
            "Upload gambar",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for f in uploaded_files:
                images.append(Image.open(f).convert("RGB"))
    else:
        camera = st.camera_input("Ambil gambar")
        if camera:
            images.append(Image.open(camera).convert("RGB"))
    summary = {
        "healthy": 0,
        "citrus_canker": 0,
        "melanose": 0,
        "tidak_valid": 0
    }
    if images:
        for i, image in enumerate(images):
            st.markdown("---")
            col1, col2 = st.columns([1,1])
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(
                    image,
                    caption=f"Gambar {i+1}",
                    width=350
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="result-card">
                """, unsafe_allow_html=True)
                img = preprocess(image)
                pred = model(tf.constant(img))
                pred = list(pred.values())[0].numpy()[0]
                idx_pred = np.argmax(pred)
                result = class_names[idx_pred]
                confidence = float(np.max(pred))
                st.markdown("""
                <div class="result-title">
                    🔍 Hasil Prediksi
                </div>
                """, unsafe_allow_html=True)
                if confidence < threshold:

                    st.warning("⚠️ Model tidak yakin terhadap gambar ini")
                    summary["tidak_valid"] += 1
                else:
                    summary[result] += 1
                    if result == "healthy":
                        st.markdown("""
                        <div class="badge badge-green">
                            🟢 Healthy
                        </div>
                        """, unsafe_allow_html=True)
                    elif result == "melanose":
                        st.markdown("""
                        <div class="badge badge-yellow">
                            🟡 Melanose
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="badge badge-red">
                            🔴 Citrus Canker
                        </div>
                        """, unsafe_allow_html=True)
                st.markdown(f"""
                <div class="conf-box">
                    📊 Tingkat Keyakinan Model:
                    <b>{confidence*100:.2f}%</b>
                    <br><br>
                    🎯 Kategori:
                    <b>{confidence_label(confidence)}</b>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📈 Probabilitas Setiap Kelas")
                for j, cls in enumerate(class_names):
                    persen = float(pred[j]) * 100
                    if cls == "healthy":
                        label = "🟢 Healthy"
                    elif cls == "melanose":
                        label = "🟡 Melanose"
                    else:
                        label = "🔴 Citrus Canker"
                    st.markdown(f"""
                    <div class="prob-label">
                        {label} — {persen:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(float(pred[j]), 1.0))
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("## 📊 Ringkasan")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🟢 Healthy", summary["healthy"])
        col2.metric("🔴 Canker", summary["citrus_canker"])
        col3.metric("🟡 Melanose", summary["melanose"])
        col4.metric("⚠️ Tidak Valid", summary["tidak_valid"])
# =========================
# PANDUAN
# =========================
elif menu == "Panduan":
    st.markdown("""
    <div class="title">
         Panduan Penggunaan
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="subtitle">
        Cara menggunakan sistem deteksi penyakit buah jeruk
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h2> 1. Pilih Gambar</h2>
    <ul>
        <li>Masuk ke menu Deteksi</li>
        <li>Upload gambar atau gunakan kamera</li>
        <li>Pastikan gambar jelas dan fokus</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h2> 2. Proses Deteksi AI</h2>
    <p>
    Sistem akan memproses gambar menggunakan
    model Deep Learning CNN MobileNetV2.
    </p>
    <ul>
        <li>🟢 Healthy</li>
        <li>🟡 Melanose</li>
        <li>🔴 Citrus Canker</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
    <h2> Tips Gambar yang Bagus</h2>
    <ul>
        <li>Gunakan pencahayaan yang cukup</li>
        <li>Pastikan gambar tidak blur</li>
        <li>Fokus pada buah jeruk</li>
        <li>Gunakan resolusi gambar yang jelas</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
# =========================
# ABOUT
# =========================
elif menu == "About":
    st.markdown("""
    <div class="title">
        Tentang Sistem deteksi
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h2> Tentang Sistem</h2>
    <p>
    Website ini digunakan untuk mendeteksi penyakit pada buah jeruk
    berdasarkan citra digital menggunakan model Convolutional Neural Network (CNN).
    </p>
    <p>
    Sistem mampu mengklasifikasikan:
    </p>
    <ul>
        <li>🍊 Citrus Canker</li>
        <li>🍊 Melanose</li>
        <li>🍊 Healthy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
    <h2>🔴 Citrus Canker</h2>
    <p>
    Citrus Canker merupakan penyakit pada tanaman jeruk yang disebabkan
    oleh bakteri <b>Xanthomonas citri</b>.
    </p>
    <ul>
        <li>Muncul bercak coklat atau kehitaman</li>
        <li>Permukaan buah terlihat kasar</li>
        <li>Terdapat lingkaran kekuningan pada bercak</li>
        <li>Buah mudah rusak dan kualitas menurun</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
    <h2>🟡 Melanose</h2>
    <p>
    Melanose adalah penyakit jamur pada buah jeruk
    yang disebabkan oleh jamur <b>Diaporthe citri</b>.
    </p>
    <ul>
        <li>Muncul bercak hitam kecil</li>
        <li>Permukaan kulit buah menjadi kasar</li>
        <li>Bercak menyebar pada kulit buah</li>
        <li>Mengurangi kualitas buah jeruk</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
    <h2>🟢 Healthy</h2>
    <p>
    Healthy menunjukkan kondisi buah jeruk yang sehat
    dan tidak terindikasi penyakit.
    </p>
    <ul>
        <li>Permukaan kulit halus dan bersih</li>
        <li>Tidak terdapat bercak hitam atau coklat</li>
        <li>Warna buah segar dan merata</li>
        <li>Tidak terdapat kerusakan pada kulit</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
    <h2> Model Deep Learning</h2>
    <p>
    Sistem menggunakan teknologi
    <b>Convolutional Neural Network (CNN)</b>
    dengan arsitektur
    <b>MobileNetV2</b>.
    </p>
    <ul>
        <li>🔴 Citrus Canker</li>
        <li>🟡 Melanose</li>
        <li>🟢 Healthy</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
 # =========================
# KONTAK
# =========================
elif menu == "Kontak":

    st.markdown("""
    <div class="title">
         Kontak
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="
        width:70%;
        margin:auto;
        margin-top:40px;
        text-align:left;
        font-size:24px;
        line-height:2;
    ">
    <b>Email</b> : aditiaprasojo123@gmail.com <br>
    <b>Program Studi</b> : Teknik Informatika <br>
    <b>Universitas</b> : universitas Dian Nuswantoro <br><br>
    <div style="
        background: rgba(34,197,94,0.15);
        padding:18px;
        border-radius:18px;
        text-align:center;
        font-size:22px;
        color:#86efac;
        border:1px solid rgba(34,197,94,0.3);
    ">
         Terima kasih telah menggunakan sistem ini
    </div>
    </div>
    """, unsafe_allow_html=True)