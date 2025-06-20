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

# === KONFIGURASI HALAMAN === #
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="centered", page_icon="👁️")
st.markdown("""
    <style>
        .stApp {background: linear-gradient(to bottom right, #e0f7fa, #fce4ec);}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        .sidebar .sidebar-content {background-color: #bbdefb;}
    </style>
""", unsafe_allow_html=True)

# === LOAD MODEL DAN LABEL MAP === #
@st.cache_resource
def load_model():
    model_path = os.path.join("saved_model")
    return tf.saved_model.load(model_path)

@st.cache_data
def load_label_map():
    with open("label_map.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
label_map = load_label_map()
label_inv_map = {v: k for k, v in label_map.items()}

# === NORMALISASI LABEL === #
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

# === REKOMENDASI PER PENYAKIT === #
rekomendasi = {
    "cataract": "Mata terdeteksi cataract.\n- Gunakan kacamata untuk membantu penglihatan sementara.\n- Hindari aktivitas di bawah cahaya terang tanpa pelindung.\n- Pertimbangkan operasi katarak bila kualitas hidup terganggu.\n- Konsultasi rutin ke dokter mata untuk pemantauan.",
    "glaucoma": "Mata terdeteksi glaucoma.\n- Segera konsultasi ke dokter mata untuk evaluasi tekanan bola mata.\n- Gunakan obat tetes mata yang diresepkan secara teratur.\n- Hindari stres dan aktivitas yang meningkatkan tekanan bola mata.\n- Pemeriksaan berkala untuk mencegah kerusakan saraf mata.",
    "retina disease": "Mata terdeteksi penyakit retina.\n- Periksakan ke dokter spesialis retina secepatnya.\n- Pemeriksaan OCT/angiografi retina disarankan.\n- Pengobatan mungkin mencakup laser, injeksi, atau operasi kecil.\n- Deteksi dini penting untuk mencegah kehilangan penglihatan permanen.",
    "normal": "Mata terdeteksi normal.\n- Jaga kesehatan mata dengan pola hidup sehat.\n- Istirahatkan mata dari layar setiap 20 menit (aturan 20-20-20).\n- Konsumsi makanan kaya vitamin A, C, dan E.\n- Lakukan pemeriksaan mata rutin tiap 6–12 bulan."
}

# === CLEAN TEKS === #
def clean_text_for_pdf(text):
    replacements = {"–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "•": "-", "\u202f": " "}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "ignore").decode("latin-1")

# === PREDIKSI === #
def predict_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    serving_fn = model.signatures["serving_default"]
    output = serving_fn(tf.constant(img_array))
    output_key = list(output.keys())[0]
    predictions = output[output_key].numpy()
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    return label_inv_map[class_index], confidence

# === GENERATE PDF === #
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

# === SIMPAN RIWAYAT === #
def simpan_riwayat(email, label, confidence):
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[email, label, confidence, waktu]], columns=["Email", "Label", "Confidence", "Waktu"])
    if os.path.exists("riwayat.csv"):
        df.to_csv("riwayat.csv", mode='a', index=False, header=False)
    else:
        df.to_csv("riwayat.csv", index=False)

# === RESET RIWAYAT === #
def reset_riwayat():
    if os.path.exists("riwayat.csv"):
        os.remove("riwayat.csv")
        st.success("✅ Riwayat berhasil dihapus.")
    else:
        st.info("ℹ️ Tidak ada data riwayat yang bisa dihapus.")

# === SIDEBAR MENU === #
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
            label, confidence = predict_image(image)
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
        encoded_msg = base64.urlsafe_b64encode(f"{share_text} - {share_url}".encode()).decode()
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
