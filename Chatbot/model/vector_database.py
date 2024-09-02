import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import faiss
from transformers import BartTokenizer, BartModel

# Configuración de API de Hugging Face
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# Inicialización de modelos y tokenizadores
model_name = 'facebook/bart-large'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartModel.from_pretrained(model_name)

# Leer el archivo CSV
def load_data_from_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(df.columns)  # Verifica las columnas
        questions = df['Pregunta'].tolist()
        answers = df['Respuesta'].tolist()
        return questions, answers
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo CSV: {e}")

# Codificar preguntas en vectores
def encode_questions(questions):
    try:
        inputs = tokenizer(questions, return_tensors='pt', padding=True, truncation=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    except Exception as e:
        raise RuntimeError(f"Error al codificar preguntas: {e}")

# Construcción del índice FAISS
def build_faiss_index(questions):
    try:
        question_vectors = encode_questions(questions)
        index = faiss.IndexFlatL2(question_vectors.shape[1])
        index.add(question_vectors)
        return index
    except Exception as e:
        raise RuntimeError(f"Error al construir el índice FAISS: {e}")

# Cargar datos y construir índice
csv_path = 'faq.csv'  # Ruta al archivo CSV
questions, answers = load_data_from_csv(csv_path)
index = build_faiss_index(questions)

# Búsqueda en FAISS
def search(query):
    try:
        query_vector = encode_questions([query])
        D, I = index.search(query_vector, k=1)
        return answers[I[0][0]]
    except Exception as e:
        return f"Lo siento, ocurrió un error durante la búsqueda: {e}"
