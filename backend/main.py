import os
import base64

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =====================================================
# MODELOS
# =====================================================

class ImageRequest(BaseModel):
    image: str

# =====================================================
# FUNCIÓN GEMINI
# =====================================================

def analizar(imagen_b64, prompt):

    imagen_bytes = base64.b64decode(
        imagen_b64.split(",")[1]
    )

    respuesta = (
        client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=imagen_bytes,
                    mime_type="image/jpeg"
                ),
                prompt
            ]
        )
    )

    return respuesta.text

# =====================================================
# AYUDA AL CAMINAR
# =====================================================

@app.post("/environment")

def environment(data: ImageRequest):

    prompt = """
Actúa como guía para una persona ciega.

Describe:

- obstáculos
- personas
- puertas
- escaleras
- objetos cercanos

Máximo 25 palabras.

Prioriza seguridad.
"""

    texto = analizar(
        data.image,
        prompt
    )

    return {
        "result": texto
    }

# =====================================================
# OCR
# =====================================================

@app.post("/ocr")

def ocr(data: ImageRequest):

    prompt = """
Lee todo el texto visible, evitando texto de fondo, marcas de agua o elementos decorativos.
Conserva el orden de lectura, los saltos de línea, la puntuación y las etiquetas o encabezados presentes.
"""

    texto = analizar(
        data.image,
        prompt
    )

    return {
        "result": texto
    }

# =====================================================
# DINERO
# =====================================================

@app.post("/money")

def money(data: ImageRequest):

    prompt = """
Identifica cuidadosamente billetes y monedas visibles.
Si la imagen no permite una identificación confiable, responde "No se puede determinar con certeza" en lugar de adivinar.
Responde únicamente:

valor total y moneda.
"""

    texto = analizar(
        data.image,
        prompt
    )

    return {
        "result": texto
    }

# =====================================================
# Reconocer Objetos
# =====================================================

@app.post("/object")

def object(data: ImageRequest):

    prompt = """
Describe solamente el objeto que se encuentra en la imagen en primer plano, ignora el fondo personas y objetos secundarios.
Usa solamente 50 palabras.
"""

    texto = analizar(
        data.image,
        prompt
    )

    return {
        "result": texto
    }

# =====================================================
# FRONTEND
# =====================================================

app.mount(
    "/static",
    StaticFiles(directory="../frontend"),
    name="static"
)


@app.get("/")
async def home():
    return FileResponse("../frontend/index.html")
