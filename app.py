import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

model = load_model('/content/drive/MyDrive/models/final_model.keras')

class_names = {0: 'PNEUMONIA', 1: 'NORMAL'}

st.title("🩺 Pneumonia & COVID-19 Detection from X-ray Images")
st.write("This tool helps in detecting pneumonia, including COVID-19-induced pneumonia, using deep learning models trained on X-ray images. Early detection can assist in better diagnosis and treatment.")
st.write("### 🏥 About Pneumonia")
st.write("Pneumonia is an infection that inflames the air sacs in one or both lungs. Symptoms include cough, fever, and difficulty breathing. Detecting pneumonia early is crucial for effective treatment, especially in cases related to COVID-19.")

uploaded_file = st.file_uploader("📂 Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='📷 Uploaded Image', use_column_width=True)

    img = img.convert("RGB") 
    img = img.resize((224, 224))  
    img_array = np.array(img)  

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  

    with st.spinner("🧐 Analyzing the X-ray... Please wait..."):
        time.sleep(2)
        prediction = model.predict(img_array)

    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)
    
    if prediction.shape[1] == 1:  
        pneumonia_prob = prediction[0][0]  
        normal_prob = 1 - pneumonia_prob  
    else:  
        pneumonia_prob = prediction[0][0] 
        normal_prob = prediction[0][1]  

    if pneumonia_prob > normal_prob:
        predicted_class_name = class_names[0]  
        max_prob = pneumonia_prob
        st.error(f"❌ **Predicted class:** {predicted_class_name} (High Risk)")
        st.write("⚠️ This X-ray suggests pneumonia, which could be caused by bacterial or viral infections, including COVID-19. Please consult a medical professional for further evaluation.")
    else:
        predicted_class_name = class_names[1]  
        max_prob = normal_prob
        st.success(f"✅ **Predicted class:** {predicted_class_name} (Healthy)")
        st.snow()
        st.write("🎉 No signs of pneumonia detected! However, always consult with a doctor for an accurate medical assessment.")

    st.write(f"### 🔬 **Prediction Results**")
    st.write(f"**Probability:** {max_prob:.4f}")
    st.write("---")
    st.write(f"📊 **Probability for PNEUMONIA:** {pneumonia_prob:.4f}")
    st.write(f"📊 **Probability for NORMAL:** {normal_prob:.4f}")

st.write("### 📌 Disclaimer")
st.write("This AI model is a supportive tool and should not replace professional medical diagnosis. Always consult with a healthcare provider for an accurate assessment.")
