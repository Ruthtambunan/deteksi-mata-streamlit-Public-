import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import datetime
from fpdf import FPDF
import base64
import yagmail
from streamlit_option_menu import option_menu
import os

# Fungsi memuat model TFLite
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return image_array.astype(np.float32).reshape(1, 224, 224, 3)

# Fungsi prediksi
def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = float(np.max(output_data))
    return prediction, confidence

# Fungsi klasifikasi label
def get_class_name(pred):
    classes = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
    return classes[pred]

# Fungsi membuat laporan PDF
def generate_pdf(image, result_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hasil Deteksi Penyakit Mata", ln=True, align='C')
    pdf.ln(10)

    image_path = "temp_image.jpg"
    image.save(image_path)
    pdf.image(image_path, x=60, y=30, w=90)
    pdf.ln(85)

    pdf.multi_cell(0, 10, txt=result_text)
    os.remove(image_path)

    pdf_output = "hasil_deteksi.pdf"
    pdf.output(pdf_output)
    return pdf_output

# Fungsi kirim email
def send_email(receiver_email, attachment_path):
    try:
        yag = yagmail.SMTP(user=os.environ.get("EMAIL_USER"), password=os.environ.get("EMAIL_PASS"))
        yag.send(
            to=receiver_email,
            subject="Hasil Deteksi Penyakit Mata",
            contents="Berikut terlampir hasil deteksi dari aplikasi.",
            attachments=attachment_path,
        )
        return True
    except Exception as e:
        st.error(f"Email gagal dikirim: {e}")
        return False

# UI
def main():
    st.set_page_config(page_title="Deteksi Penyakit Mata", layout="centered")

    with st.sidebar:
        selected = option_menu(
            "Menu",
            ["Beranda", "Deteksi", "Tentang"],
            icons=["house", "activity", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    if selected == "Beranda":
        st.title("Selamat datang di Aplikasi Deteksi Penyakit Mata")
        st.write("Aplikasi ini menggunakan model pembelajaran mesin untuk mendeteksi jenis penyakit mata dari gambar retina.")
        st.image("assets/eye.jpg", use_column_width=True)

    elif selected == "Deteksi":
        st.title("Deteksi Penyakit Mata")
        uploaded_file = st.file_uploader("Unggah gambar retina", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_column_width=True)

            model = load_model()
            processed = preprocess_image(image)
            pred_class, confidence = predict_image(model, processed)
            class_name = get_class_name(pred_class)

            st.success(f"Hasil Deteksi: **{class_name}**")
            st.info(f"Akurasi Prediksi: {confidence * 100:.2f}%")

            now = datetime.datetime.now()
            result_text = f"""
            Hasil Deteksi Penyakit Mata
            ============================
            Tanggal: {now.strftime('%d-%m-%Y %H:%M:%S')}
            Hasil: {class_name}
            Akurasi: {confidence * 100:.2f}%
            """

            st.text_area("Ringkasan Hasil:", result_text, height=200)

            if st.button("Unduh Laporan PDF"):
                pdf_path = generate_pdf(image, result_text)
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pdf_path}">Klik di sini untuk mengunduh laporan</a>'
                    st.markdown(href, unsafe_allow_html=True)

            email_input = st.text_input("Kirim hasil ke email (opsional):")
            if email_input and st.button("Kirim Email"):
                pdf_path = generate_pdf(image, result_text)
                if send_email(email_input, pdf_path):
                    st.success("Email berhasil dikirim.")

    elif selected == "Tentang":
        st.title("Tentang Aplikasi")
        st.markdown("""
        Aplikasi ini dikembangkan untuk membantu deteksi dini penyakit mata seperti:
        - Katarak
        - Glaukoma
        - Retinopati Diabetik

        Dibuat menggunakan:
        - Streamlit
        - TensorFlow Lite
        - Model klasifikasi citra

        Kontak: developer@example.com
        """)

if __name__ == "__main__":
    main()
