import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template
from controller.chatbot_controller import chatbot_blueprint


# Especificar nuevas rutas para archivos estáticos y plantillas
app = Flask(__name__, 
            static_folder=os.path.join('view', 'static'),
            template_folder=os.path.join('view', 'templates'))

# Registrar el blueprint
app.register_blueprint(chatbot_blueprint)

# Ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)
