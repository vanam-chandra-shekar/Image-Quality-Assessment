
import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image, ImageStat, ImageEnhance
import io
import pandas as pd
import base64

st.set_page_config(page_title="Image Quality Assessment", layout="centered")
st.title("📷 Image Quality Assessment (MOS Prediction)")
st.markdown("Upload an image to get its predicted **Quality Rating** and see its quality on a visual scale.")

with st.expander("📘 About this App", expanded=False):
    st.markdown("""
This app uses a **deep learning model** to predict image quality on a scale of 1–10, based on MOS (Mean Opinion Score).

**Steps:**
1. Upload image
2. View brightness, contrast, sharpness, colorfulness
3. Apply manual or auto enhancements
4. View enhanced preview + download
5. Predict quality with visual scale
""")

st.sidebar.title("Enhancement Options")
enhancement_mode = st.sidebar.radio("Enhancement Mode", ["Manual", "Auto", "Off"], index=0)

brightness_slider = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
contrast_slider = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
sharpness_slider = st.sidebar.slider("Sharpness", 0.5, 2.0, 1.0, 0.1)
color_slider = st.sidebar.slider("Colorfulness", 0.5, 2.0, 1.0, 0.1)

uploaded_file = st.file_uploader("📤 Choose an image", type=["jpg", "jpeg", "png"])

def calculate_brightness(img):
    stat = ImageStat.Stat(img.convert("L"))
    return stat.mean[0]

def calculate_contrast(img):
    stat = ImageStat.Stat(img.convert("L"))
    return stat.stddev[0]

def calculate_sharpness(img):
    arr = np.array(img.convert("L"), dtype='int32')
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2).mean()

def calculate_colorfulness(img):
    img_np = np.array(img).astype("float")
    (R, G, B) = (img_np[:,:,0], img_np[:,:,1], img_np[:,:,2])
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return np.sqrt(np.std(rg)**2 + np.std(yb)**2) + 0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)

def interpret_quality(value, thresholds, labels, suggestions):
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return labels[i], suggestions[i]
    return labels[-1], suggestions[-1]

def get_auto_enhancement_factors(brightness, contrast, sharpness, colorfulness):
    b = 1.3 if brightness < 100 else (0.8 if brightness > 180 else 1.0)
    c = 1.4 if contrast < 30 else 1.0
    s = 1.5 if sharpness < 1 else 1.0
    col = 1.4 if colorfulness < 15 else 1.0
    return b, c, s, col

if uploaded_file:
    temp_image_path = "temp_image.jpg"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    img = Image.open(temp_image_path)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    brightness = calculate_brightness(img)
    contrast = calculate_contrast(img)
    sharpness = calculate_sharpness(img)
    colorfulness = calculate_colorfulness(img)

    st.subheader("📊 Basic Image Quality Attributes")
    brightness_label, brightness_tip = interpret_quality(brightness, [100, 200], ["Low", "Normal", "High"], ["Increase brightness.", "Looks good.", "Too bright."])
    contrast_label, contrast_tip = interpret_quality(contrast, [30], ["Low", "Good"], ["Increase contrast.", "Looks good."])
    sharpness_label, sharpness_tip = interpret_quality(sharpness, [1], ["Low", "Sharp"], ["Sharpen the image.", "Looks sharp."])
    colorfulness_label, colorfulness_tip = interpret_quality(colorfulness, [10], ["Low", "Vibrant"], ["Increase saturation.", "Great colors."])

    quality_summary = pd.DataFrame({
        "Metric": ["Brightness", "Contrast", "Sharpness", "Colorfulness"],
        "Value": [f"{brightness:.2f}", f"{contrast:.2f}", f"{sharpness:.2f}", f"{colorfulness:.2f}"],
        "Interpretation": [brightness_label, contrast_label, sharpness_label, colorfulness_label],
        "Suggestion": [brightness_tip, contrast_tip, sharpness_tip, colorfulness_tip]
    })
    st.dataframe(quality_summary, hide_index=True, use_container_width=True)

    # Enhancements
    enhanced_img = img.copy()

    if enhancement_mode != "Off":
        st.subheader("✨ Enhancement Preview")
        if enhancement_mode == "Auto":
            b, c, s, col = get_auto_enhancement_factors(brightness, contrast, sharpness, colorfulness)
            st.markdown(f"🔧 **Auto Applied**  \n- Brightness: `{b}`  \n- Contrast: `{c}`  \n- Sharpness: `{s}`  \n- Colorfulness: `{col}`")
        else:
            b, c, s, col = brightness_slider, contrast_slider, sharpness_slider, color_slider

        for enhancer_class, factor in zip(
            [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Sharpness, ImageEnhance.Color],
            [b, c, s, col]
        ):
            enhanced_img = enhancer_class(enhanced_img).enhance(factor)

        st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

        buf = io.BytesIO()
        enhanced_img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f'<a href="data:image/jpeg;base64,{b64}" download="enhanced_image.jpg">📥 Download Enhanced Image</a>', unsafe_allow_html=True)
    else:
        st.info("Enhancements are disabled.")

    # Overwrite temp image for prediction
    enhanced_img.save(temp_image_path)

    # Prediction directly after preview
    with open(temp_image_path, "rb") as img_file:
        response = requests.post("https://image-quality-assessment-1.onrender.com", files={"file": img_file})


    if response.status_code == 200:
        result = response.json()
        mos = result.get("MOS_Prediction", None)

        if mos is not None:
            mos_10scale = (mos - 1.096) / (4.31 - 1.096) * 9 + 1
            st.subheader(f"📈 Predicted Quality Rating: `{mos_10scale:.2f}` / 10")
            st.markdown(f"**Raw MOS (1–5 scale):** `{mos:.2f}`")

            def interpret_mos(mos_value):
                if mos_value < 1.8:
                    return "Very Poor", "🔴"
                elif mos_value < 2.5:
                    return "Poor", "🟠"
                elif mos_value < 3.3:
                    return "Average", "🟡"
                elif mos_value < 4:
                    return "Good", "🟢"
                else:
                    return "Excellent", "✅"

            interpretation, emoji = interpret_mos(mos)
            st.markdown(f"**{emoji} Quality Interpretation:** {interpretation}")

            fig, ax = plt.subplots(figsize=(8, 1.2))
            cmap = plt.get_cmap("RdYlGn")
            norm_val = (mos_10scale - 1) / 9
            ax.barh([0], [mos_10scale], color=cmap(norm_val), height=0.5)
            ax.set_xlim(1, 10)
            ax.set_xticks(np.arange(1, 11, 1))
            ax.set_yticks([])
            ax.set_title("Quality Rating Scale (1 to 10)", fontsize=10)
            ax.axvline(mos_10scale, color='black', linestyle='--', linewidth=1)
            st.pyplot(fig)
        else:
            st.error(" Couldn't retrieve MOS prediction.")
    else:
        st.error(f"API Error: {response.status_code} - {response.text}")

