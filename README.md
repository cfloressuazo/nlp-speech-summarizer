# Tarea De Procesamiento de Lenguaje Natural
Integrantes:
- Cesar Flores
- Juan Francisco Pinto
- Fernando Garcia

El código de la applicación se encuentra comentado (en inglés) en el archivo `app.py` de esta carpeta. El código posee múltiples funciones que están encargadas en procesar la carpeta `data` con los discursos, transformar estos en formato para entrenar un modelo de embedding `Word2Vec` y luego para procesar cualquier archivo o/u discurso para generar un resumen de este a través de la función `create_summary`.

Se entrenaron dos modelos de prueba con distintos parametros, los que se pueden probar para generar el resumen y comparar. Para esto debes cargar el modelo con la función `load_w2v_model` con el path del modelo (i.e. `models/300-5-w2v.model`).


# Instalar
1. Crea un ambiente virtual de python `python -m venv .venv`.
2. Activalo `. .venv/bin/activate`
3. Instala las dependencias `pip install -r requirements.txt`

# Ejecucción
1. Puedes ver un ejemplo interactivo en el `interactive_notebook.ipynb`
2. A traves de la terminal, ejecuta:
   1. Para entrenar un model: `python app.py -m 'models/300-5-w2v.model'`
   2. Para generar un resumen: `python app.py -s 'data/DiscursosOriginales/72320.txt'`
