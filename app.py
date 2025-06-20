# === STRUKTUR FOLDER YANG VALID UNTUK STREAMLIT CLOUD ===

# project-root/
# ├── app.py
# ├── requirements.txt
# ├── label_map.pkl
# └── tflite_model/
#     └── eye_disease_model.tflite

# PASTIKAN FILE INI (app.py) DI ROOT DAN TFLITE FILE ADA DI FOLDER `tflite_model`.

import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import time
from fpdf import FPDF
import yagmail
import os
import pandas as pd
from datetime import datetime
import base64
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Deteksi Penyakit Mata", layout="centered", page_icon="👁️")
st.markdown("""
    <style>
        .stApp {background: linear-gradient(to bottom right, #e0f7fa, #fce4ec);}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .sidebar .sidebar-content {background-color: #bbdefb;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="tflite_model/eye_disease_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_label_map():
    with open("label_map.pkl", "rb") as f:
        return pickle.load(f)

interpreter = load_tflite_model()
label_map = load_label_map()
label_inv_map = {v: k for k, v in label_map.items()}

label_mapping = {
    "retina_desiese": "retina disease",
    "retina disease": "retina disease",
    "retinadesiese": "retina disease",
    "retina_disease": "retina disease",
    "cataract": "cataract",
    "catarak": "cataract",
    "katarak": "cataract",
    "glaucoma": "glaucoma",
    "glaukoma": "glaucoma",
    "normal": "normal",
    "mata normal": "normal"
}

rekomendasi = {
    "cataract": "Mata terdeteksi cataract.\n- Gunakan kacamata...",
    "glaucoma": "Mata terdeteksi glaucoma.\n- Segera konsultasi...",
    "retina disease": "Mata terdeteksi penyakit retina.\n- Periksakan ke dokter...",
    "normal": "Mata terdeteksi normal.\n- Jaga kesehatan mata..."
}

def clean_text_for_pdf(text):
    replacements = {"–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "•": "-", "\u202f": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")

def predict_image_tflite(interpreter, img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype("float32")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]
    return label_inv_map[class_index], confidence

def generate_pdf(label, confidence, rekom, img):
    rekom_clean = clean_text_for_pdf(rekom)
    label_clean = clean_text_for_pdf(label)
    img_path = "gambar_uploaded.jpg"
    img.save(img_path)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font_size(16)
    pdf.cell(200, 10, txt="Hasil Deteksi Penyakit Mata", ln=True, align='C')
    pdf.ln(10)
    pdf.image(img_path, x=30, y=40, w=150)
    pdf.ln(100)
    pdf.set_font_size(12)
    pdf.multi_cell(0, 10, f"Hasil Deteksi: {label_clean}\nKeyakinan: {confidence:.2f}%\n\nRekomendasi:\n{rekom_clean}")
    pdf.output("hasil_deteksi.pdf")

def simpan_riwayat(email, label, confidence):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[email, label, confidence, waktu]], columns=["Email", "Label", "Confidence", "Waktu"])
    if os.path.exists("riwayat.csv"):
        df.to_csv("riwayat.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("riwayat.csv", index=False)

def reset_riwayat():
    if os.path.exists("riwayat.csv"):
        os.remove("riwayat.csv")
        st.success("✅ Riwayat berhasil dihapus.")
    else:
        st.info("ℹ️ Tidak ada data riwayat yang bisa dihapus.")

with st.sidebar:
    menu = option_menu("Menu", ["Deteksi Mata", "Riwayat Deteksi"],
                       icons=["eye", "clock-history"], menu_icon="cast", default_index=0)

if menu == "Deteksi Mata":
    st.markdown("<h1 style='text-align:center;'>🔬 Aplikasi Deteksi Penyakit Mata</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Gambar Mata", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="🖼️ Gambar yang Diunggah", use_container_width=True)

        with st.spinner("⏳ Sedang memproses..."):
            time.sleep(1)
            label, confidence = predict_image_tflite(interpreter, image)
            confidence_percent = confidence * 100
            standard_label = label_mapping.get(label.lower().strip(), "unknown")
            rekom = rekomendasi.get(standard_label, "- Rekomendasi tidak tersedia.\n- Silakan konsultasi ke dokter mata.")
            rekom_html = rekom.replace("\n", "<br>")

        st.markdown(f"""
        <table class="custom-table">
            <tr><th>🔍 Deteksi</th><td><b>{label}</b></td></tr>
            <tr><th>📊 Keyakinan</th><td>{confidence_percent:.2f}%</td></tr>
            <tr><th>🩺 Rekomendasi</th><td>{rekom_html}</td></tr>
        </table>
        """, unsafe_allow_html=True)

        st.caption(f"🧪 Label asli dari model: {label} → Mapping ke: {standard_label}")
        generate_pdf(label, confidence_percent, rekom, image)
        with open("hasil_deteksi.pdf", "rb") as f:
            st.download_button("📄 Unduh PDF", f, file_name="hasil_deteksi.pdf", mime="application/pdf")

        email_input = st.text_input("📧 Masukkan Email untuk Kirim Hasil:")
        if st.button("Kirim Email"):
            if email_input:
                try:
                    yag = yagmail.SMTP("ruthoktatambunan@gmail.com", "ybzu tmnz tyqb twoj")
                    yag.send(
                        to=email_input,
                        subject="Hasil Deteksi Penyakit Mata",
                        contents="Berikut terlampir hasil deteksi mata Anda.",
                        attachments="hasil_deteksi.pdf"
                    )
                    simpan_riwayat(email_input, label, confidence_percent)
                    st.success("✅ Email dan riwayat berhasil disimpan.")
                except Exception as e:
                    st.error(f"❌ Gagal mengirim email: {e}")
            else:
                st.warning("⚠️ Masukkan alamat email terlebih dahulu.")

        st.subheader("🔗 Bagikan ke Media Sosial")
        share_text = f"Hasil deteksi: {label} ({confidence_percent:.2f}%)"
        share_url = "https://github.com/Ruthtambunan/Aplikasi_deteksi_penyakit_mata"
        st.markdown(f'''
        - [💬 WhatsApp](https://wa.me/?text={share_text} - {share_url})
        - [📘 Facebook](https://www.facebook.com/sharer/sharer.php?u={share_url})
        - [🐦 Twitter](https://twitter.com/intent/tweet?text={share_text}&url={share_url})
        ''')

elif menu == "Riwayat Deteksi":
    st.subheader("📜 Riwayat Deteksi Anda")
    if os.path.exists("riwayat.csv"):
        df = pd.read_csv("riwayat.csv", names=["Email", "Label", "Confidence", "Waktu"], header=None)
        st.dataframe(df.sort_values(by="Waktu", ascending=False))
        if st.button("🗑️ Hapus Semua Riwayat"):
            reset_riwayat()
    else:
        st.info("Belum ada data riwayat.")
