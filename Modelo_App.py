
import streamlit as st
import pickle
import joblib
from lime.lime_text import LimeTextExplainer

# Cargar el modelo y el vectorizador entrenados
svm_model = joblib.load('C:/Users/Luis.fernandez/01. Python/13. Proyecto ML y NLP/03. Modelo Final/modelo_entrenado_comprimido.pkl')

with open('C:/Users/Luis.fernandez/01. Python/13. Proyecto ML y NLP/03. Modelo Final/vectorizador_entrenado.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Instanciar LimeTextExplainer
explainer = LimeTextExplainer(class_names=['HAM', 'SPAM'])

# Configuración de la aplicación Streamlit
st.title("Detector de Spam")
st.write("Ingrese un mensaje y presione el botón para verificar si es spam o no.")

# Entrada de texto del usuario
input_text = st.text_area("Escriba su mensaje aquí:")

# Función para realizar la predicción e interpretación
def interpretar_mensaje(mensaje):
    # Transformar el mensaje usando el vectorizador
    vector_mensaje = tfidf.transform([mensaje]).toarray()
    # Obtener la explicación de LIME
    exp = explainer.explain_instance(
        mensaje,
        lambda x: svm_model.predict_proba(tfidf.transform(x).toarray()),
        num_features=10
    )
    return exp

# Botón para ejecutar la verificación de spam
if st.button("Verificar si es Spam"):
    if input_text:
        # Ejecutar la interpretación
        explicacion = interpretar_mensaje(input_text)

        # Mostrar resultado de la predicción
        st.write("Resultado y explicación:")
        st.components.v1.html(explicacion.as_html(), height=800)
    else:
        st.write("Por favor, ingrese un mensaje para analizar.")
