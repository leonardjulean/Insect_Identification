import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import os

# Configurarea paginii
st.set_page_config(
    page_title="Recunoaștere Insecte",
    page_icon="🐝",
    layout="wide"
)

# Funcție de încărcare model
@st.cache_resource
def load_custom_model(model_path):
    """Load trained model and retrieve class names."""
    model = load_model(model_path)
    
    # Get input size dynamically from the model
    input_shape = model.input_shape[1:3]  # Extract (height, width)
    
    # Get number of classes from output layer
    num_classes = model.output_shape[1]
    
    return model, input_shape, num_classes

# Funcție de predicție
def predict_insect(model, img, class_names, img_size):
    """Predict insect species from an image using the provided model."""
    
    # Convert PIL image to OpenCV format
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)  # Convert back to RGB

    # Resize image based on model's input size
    img_resized = cv2.resize(img_cv2, img_size)
    
    # Preprocess for model
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 120)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Filter out very small and very large contours
        valid_contours = [cnt for cnt in contours if 500 < cv2.contourArea(cnt) < 100000]

        if valid_contours:
            # Find the best contour (biggest, but within a reasonable range)
            largest_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Ensure bounding box stays within image bounds
            img_h, img_w = img_cv2.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))

            # Draw a GREEN bounding box
            cv2.rectangle(img_cv2, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Get predicted label
    predicted_label = class_names[class_index]
    
    return img_cv2, predicted_label, confidence, prediction

# Titlu principal și descriere
st.title("🐝 Recunoaștere Insecte")
st.write("Încărcați o imagine pentru a identifica tipul de insectă")

# Sidebar pentru încărcarea modelului
st.sidebar.header("Configurare Model")

# Calea către model - folosim calea relativă
MODEL_PATH = st.sidebar.text_input(
    "Calea către modelul antrenat (.h5)", 
    value="./model/mobilenet_insect_classifier.h5"
)

# Calea către fișierul class_names.json
CLASS_NAMES_PATH = st.sidebar.text_input(
    "Calea către numele claselor (JSON)",
    value="./model/class_names.json"
)

# Verificăm dacă fișierele există și altfel încercăm să le găsim
if not os.path.exists(MODEL_PATH):
    # Încercăm să găsim orice fișier .h5 în directorul curent sau în subdirectoare
    h5_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    if h5_files:
        MODEL_PATH = h5_files[0]  # Folosim primul model găsit
        st.sidebar.info(f"Model găsit automat: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    # Încercăm să găsim fișierul class_names.json în directorul curent sau în subdirectoare
    json_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'class_names.json':
                json_files.append(os.path.join(root, file))
    
    if json_files:
        CLASS_NAMES_PATH = json_files[0]  # Folosim primul fișier găsit
        st.sidebar.info(f"Fișier class_names.json găsit automat: {CLASS_NAMES_PATH}")

# Încărcăm numele claselor din fișierul JSON
class_names = None
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        st.sidebar.success(f"✅ Nume de clase încărcate: {len(class_names)}")
    except Exception as e:
        st.sidebar.error(f"❌ Eroare la încărcarea numelor de clase: {str(e)}")
else:
    st.sidebar.warning("⚠️ Fișierul cu numele claselor nu a fost găsit. Se vor folosi nume generice.")
    # Daca nu exista fisierul, cream un demo cu nume generice
    class_names = ["Albină", "Gândac", "Fluture", "Libelulă", "Furnică", "Țânțar", "Păianjen"]
    st.sidebar.info("Se folosesc nume de clase demo pentru testare.")

# Încărcare model
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        with st.spinner("Se încarcă modelul..."):
            model, img_size, num_classes = load_custom_model(MODEL_PATH)
            
            # Dacă nu avem nume de clase, creăm nume generice
            if class_names is None:
                class_names = [f"Insecta {i+1}" for i in range(num_classes)]
            
            # Verificare dacă numărul de clase se potrivește
            if num_classes != len(class_names):
                st.sidebar.warning(f"⚠️ Atenție: Modelul are {num_classes} clase, dar lista încărcată are {len(class_names)}.")
                # Ajustare automată a listei
                if num_classes < len(class_names):
                    class_names = class_names[:num_classes]
                else:
                    class_names = class_names + [f"Insecta {i+len(class_names)+1}" for i in range(len(class_names), num_classes)]
                
            model_loaded = True
            st.sidebar.success(f"✅ Model încărcat cu succes: {os.path.basename(MODEL_PATH)}")
    except Exception as e:
        st.sidebar.error(f"❌ Eroare la încărcarea modelului: {str(e)}")
else:
    st.sidebar.warning("⚠️ Modelul nu a fost găsit. Încercați să verificați calea sau să uploadați modelul în repository.")
    # Pentru demo, oferim opțiunea de a rula în modul demo
    if st.sidebar.button("Rulare în mod demo (fără model real)"):
        st.sidebar.success("✅ Mod demo activat!")
        model_loaded = True
        # Cream un model fake pentru demo
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        model = Sequential([Dense(7, activation='softmax', input_shape=(224, 224, 3))])
        img_size = (224, 224)
        num_classes = 7
        if class_names is None or len(class_names) != num_classes:
            class_names = ["Albină", "Gândac", "Fluture", "Libelulă", "Furnică", "Țânțar", "Păianjen"]

# Zona de încărcare imagine - acum folosim o singură coloană
st.subheader("Încarcă o imagine")

# Opțiuni de încărcare
upload_method = st.radio(
    "Metodă de încărcare:", 
    options=["Încarcă fișier", "Folosește camera"]
)

if upload_method == "Încarcă fișier":
    uploaded_file = st.file_uploader("Selectați o imagine", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Fotografiază o insectă")

# Procesare imagine și predicție
if uploaded_file is not None:
    # Citire imagine
    image_pil = Image.open(uploaded_file)
    
    # Afișare imagine originală cu lățime limitată
    image_container = st.container()
    with image_container:
        image_col, _ = st.columns([2, 1])
        with image_col:
            st.image(image_pil, caption="Imagine încărcată", use_container_width=True)
    
    # Buton de analiză - mereu vizibil
    analyze_button = st.button("🔍 Identifică insecta", type="primary")
    
    if analyze_button:
        if model_loaded:
            with st.spinner("Se analizează imaginea..."):
                try:
                    # Predicție și aplicare bounding box
                    result_img, predicted_class, confidence, all_predictions = predict_insect(
                        model, image_pil, class_names, img_size
                    )
                    
                    # Card cu rezultate
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #1a405b; max-width: 600px;">
                        <h3 style="color: #1E88E5; margin-top: 0;">Insectă identificată:</h3>
                        <h2>{predicted_class}</h2>
                        <p>Grad de încredere: <b>{confidence*100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Info despre insectă
                    st.subheader("Alte predicții posibile:")
                    
                    # Sortăm predicțiile și afișăm top 3
                    sorted_predictions = sorted(
                        [(class_names[i], float(all_predictions[0][i])) for i in range(len(class_names))],
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    # Afișăm top 3 predicții
                    for i, (name, prob) in enumerate(sorted_predictions[:3]):
                        st.markdown(f"{i+1}. **{name}**: {prob*100:.2f}%")
                    
                    # Afișăm un grafic cu top 5 predicții (limitat la 600px lățime)
                    st.subheader("Top 5 predicții:")
                    top5_dict = {name: prob for name, prob in sorted_predictions[:5]}
                    chart = st.bar_chart(top5_dict)
                except Exception as e:
                    st.error(f"Eroare la analizarea imaginii: {str(e)}")
                    st.info("Verificați dacă modelul a fost încărcat corect și dacă imaginea este validă.")
        else:
            st.error("❌ Modelul nu este încărcat! Verificați calea către model.")
            st.info("Puteți folosi modul demo pentru a testa interfața.")

# Adăugare informații în sidebar
st.sidebar.header("Despre aplicație")
st.sidebar.info("""
    Această aplicație folosește învățare automată pentru a identifica
    diferite tipuri de insecte din imagini. Modelul a fost antrenat
    pentru a recunoaște mai multe specii de insecte.
    
    **Sfaturi pentru rezultate optime:**
    - Asigurați-vă că insecta este în centrul imaginii
    - Folosiți o imagine clară, bine iluminată
    - Evitați fundalurile complicate sau încărcate
""")
