import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Blueprint, request, jsonify, render_template
from model.vector_database import search
from transformers import BartTokenizer, BartForConditionalGeneration

chatbot_blueprint = Blueprint('chatbot', __name__, template_folder='view/templates')

# Configuración de BART
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Preguntas predefinidas fuera de contexto
predefined_questions = {
    "¿cuál es tu nombre?": "Soy un asistente virtual diseñado para ayudarte con información y responder a tus preguntas.",
    "¿qué eres?": "Soy un asistente virtual creado por desarrolladores humanos para proporcionar respuestas útiles basadas en los datos y el conocimiento que se me ha proporcionado.",
    "¿quién te creó?": "Fui creado por desarrolladores humanos para ayudarte con información y responder a tus preguntas.",
    "¿cuál es tu propósito?": "Mi propósito es ayudarte con información y responder a tus preguntas de la mejor manera posible.",
    "¿cómo funcionas?": "Funciono procesando las preguntas que me haces y generando respuestas basadas en los datos y el conocimiento que tengo.",
    "¿puedes aprender?": "No, no puedo aprender. Mi conocimiento está basado en los datos y la información que se me ha proporcionado.",
    "¿eres un robot?": "Soy un asistente virtual, no un robot físico, y estoy aquí para ayudarte con lo que necesites.",
    "¿puedes sentir emociones?": "No, no puedo sentir emociones. Soy solo un programa diseñado para responder preguntas.",
    "¿qué datos almacenas?": "No almaceno datos personales. Nuestra conversación es segura y se mantiene privada.",
    "¿es seguro hablar contigo?": "Sí, es seguro hablar conmigo. No almaceno datos personales y nuestra conversación es segura.",
    
    # Preguntas relacionadas con la ESPE
    "¿qué es la espe?": "La Universidad de las Fuerzas Armadas ESPE es una institución de educación superior en Ecuador que forma parte de las Fuerzas Armadas del país. Ofrece una amplia variedad de programas académicos tanto de pregrado como de posgrado.",
    "¿dónde se encuentra la universidad de las fuerzas armadas espe?": "La ESPE tiene su campus principal en Sangolquí, a las afueras de Quito, Ecuador. También cuenta con otras sedes en Latacunga y Santo Domingo.",
    "¿cuáles son las carreras que ofrece la espe?": "La ESPE ofrece diversas carreras en áreas como Ingeniería, Ciencias de la Vida, Ciencias Administrativas, Ciencias de la Computación, Ciencias Sociales, y más. Para conocer la lista completa, te recomiendo visitar el sitio web oficial de la universidad.",
    "¿cómo puedo inscribirme en la espe?": "Para inscribirte en la ESPE, debes seguir el proceso de admisión que incluye la presentación de un examen de ingreso. Puedes obtener más información sobre el proceso en el sitio web de la universidad o contactando a la oficina de admisiones.",
    "¿cuáles son los requisitos de admisión para la espe?": "Los requisitos de admisión varían según el programa académico. Generalmente, incluyen la aprobación de un examen de ingreso, además de cumplir con los requisitos académicos previos como el título de bachiller.",
    "¿la espe tiene programas de posgrado?": "Sí, la ESPE ofrece una variedad de programas de posgrado, incluyendo maestrías y doctorados en diferentes áreas del conocimiento.",
    "¿qué servicios ofrece la espe a los estudiantes?": "La ESPE ofrece una variedad de servicios a sus estudiantes, como bibliotecas, laboratorios, servicios médicos, áreas deportivas, orientación académica y apoyo psicológico.",
    "¿cuáles son las instalaciones de la espe?": "La ESPE cuenta con modernas instalaciones, incluyendo laboratorios de última generación, bibliotecas, auditorios, áreas deportivas y zonas de recreación.",
    "¿cómo contacto a la espe?": "Puedes contactar a la ESPE a través de su sitio web oficial, llamando a su central telefónica, o visitando directamente su campus en Sangolquí.",
    "¿qué modalidades de estudio ofrece la espe?": "La ESPE ofrece modalidades de estudio presencial, semipresencial y a distancia para adaptarse a las necesidades de los estudiantes."
}

@chatbot_blueprint.route('/')
def home():
    return render_template('index.html')

@chatbot_blueprint.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    query = data.get('query', '')
    answer = generate_answer(query)
    return jsonify({'answer': answer})

def generate_answer(query):
    # Respuestas predefinidas para saludos
    greetings = {
        "hola": "¡Hola! ¿Cómo puedo ayudarte hoy?",
        "buenos días": "¡Buenos días! ¿En qué puedo asistirte?",
        "buenas tardes": "¡Buenas tardes! ¿En qué puedo ayudarte?",
        "buenas noches": "¡Buenas noches! ¿Cómo puedo ayudarte?",
        "¿cómo estás?": "Estoy bien, gracias por preguntar. ¿Cómo puedo ayudarte?",
    }

    query_lower = query.lower()

    # Verificar si la pregunta coincide con alguna de las preguntas predefinidas fuera de contexto
    if query_lower in predefined_questions:
        return predefined_questions[query_lower]

    if query_lower in greetings:
        return greetings[query_lower]

    # Buscar respuesta en el CSV
    csv_answer = search(query)
    if csv_answer:
        return csv_answer

    # Si no hay respuesta en el CSV, verificar si el contexto es suficiente
    context = search(query)
    if not context or len(context) < 20:  # Si el contexto es demasiado corto
        return "Lo siento, no tengo suficiente información para responder esa pregunta en este momento."

    # Generar respuesta usando BART si hay suficiente contexto
    inputs = tokenizer.encode(context, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
    generated_answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Filtrar la respuesta generada para evitar incoherencias
    return filter_response(generated_answer)

def filter_response(response):
    unwanted_phrases = [
        "No puede ayudar a nadie.",
        "¿cómo puedes ayudarlos?",
        "La pregunta es:",
        "No hay una solución."
    ]
    for phrase in unwanted_phrases:
        if phrase in response:
            return "Lo siento, no tengo una respuesta adecuada para esa consulta."
    return response
