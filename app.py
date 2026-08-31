
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import tensorflow as tf
from PIL import Image

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Adivina Mi Dibujo",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

/* Fondo general */
.stApp {
    background: #f4f6fb;
}

/* Contenedor */
.block-container {
    max-width: 1250px;
    padding-top: 1.5rem;
}

/* Header */
.header {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 24px;
    padding: 30px 35px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(99,102,241,0.20);
}

.header h1 {
    margin: 0;
    font-size: 42px;
    font-weight: 800;
}

.header p {
    margin-top: 8px;
    font-size: 17px;
    opacity: 0.92;
}

/* Títulos */
.section-title {
    font-size: 23px;
    font-weight: 800;
    color: #25263a;
    text-align: center;
    margin-bottom: 15px;
}

/* Paneles */
.panel {
    background: white;
    border-radius: 22px;
    padding: 25px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 6px 22px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Categoría seleccionada */
.selected-box {
    background: linear-gradient(135deg, #eef2ff, #f5f3ff);
    border: 2px solid #818cf8;
    border-radius: 18px;
    padding: 18px;
    text-align: center;
    margin-bottom: 20px;
}

.selected-small {
    color: #6366f1;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
}

.selected-big {
    color: #25263a;
    font-size: 29px;
    font-weight: 800;
    margin-top: 5px;
}

/* Predicción principal */
.prediction-main {
    background: linear-gradient(135deg, #eef2ff, #faf5ff);
    border: 2px solid #c7d2fe;
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
}

.prediction-label {
    color: #6366f1;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
}

.prediction-name {
    color: #25263a;
    font-size: 38px;
    font-weight: 900;
    margin: 8px 0;
}

.prediction-confidence {
    color: #555;
    font-size: 17px;
}

/* Top 3 */
.top-card {
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 17px;
    margin-bottom: 8px;
}

.top-name {
    font-size: 17px;
    font-weight: 700;
    color: #303247;
}

.top-percent {
    float: right;
    color: #6366f1;
    font-weight: 800;
}

/* Estado */
.status {
    text-align: center;
    padding: 13px;
    border-radius: 12px;
    font-weight: 700;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #777;
    padding: 25px;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# CONFIGURACIÓN DEL MODELO
# ============================================================

CATEGORIAS = [
    "apple",
    "banana",
    "dog",
    "cat",
    "car"
]

ICONOS = {
    "apple": "🍎",
    "banana": "🍌",
    "dog": "🐶",
    "cat": "🐱",
    "car": "🚗"
}

NOMBRES = {
    "apple": "Manzana",
    "banana": "Banana",
    "dog": "Perro",
    "cat": "Gato",
    "car": "Carro"
}

# Umbral para considerar una predicción confiable
UMBRAL_CONFIANZA = 70.0

# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():

    try:

        modelo = tf.keras.models.load_model(
            "modelo_adivina_dibujo.h5"
        )

        return modelo

    except FileNotFoundError:

        st.error(
            "❌ No se encontró el archivo "
            "'modelo_adivina_dibujo.h5'."
        )

        st.info(
            "Verifica que el archivo .h5 esté en la misma "
            "carpeta que app.py."
        )

        st.stop()

    except Exception as e:

        st.error(
            "❌ Ocurrió un error al cargar el modelo."
        )

        st.code(str(e))

        st.stop()


modelo = cargar_modelo()

# ============================================================
# SESIONES
# ============================================================

if "categoria" not in st.session_state:
    st.session_state.categoria = None

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

if "terminado" not in st.session_state:
    st.session_state.terminado = False

# ============================================================
# FUNCIONES
# ============================================================

def preparar_dibujo(imagen):

    try:

        imagen = Image.fromarray(
            imagen.astype("uint8")
        )

        imagen = imagen.convert("L")

        imagen = imagen.resize((28, 28))

        imagen = np.array(
            imagen
        ).astype("float32") / 255.0

        imagen = np.expand_dims(
            imagen,
            axis=-1
        )

        imagen = np.expand_dims(
            imagen,
            axis=0
        )

        return imagen

    except Exception as e:

        st.error(
            "❌ No se pudo procesar el dibujo."
        )

        st.code(str(e))

        return None


def predecir_top3(imagen):

    try:

        imagen_preparada = preparar_dibujo(
            imagen
        )

        if imagen_preparada is None:
            return []

        probabilidades = modelo.predict(
            imagen_preparada,
            verbose=0
        )[0]

        # Verificar número de clases
        if len(probabilidades) != len(CATEGORIAS):

            st.error(
                "❌ El modelo no coincide con las "
                "5 categorías configuradas."
            )

            st.write(
                "Salidas del modelo:",
                len(probabilidades)
            )

            st.write(
                "Categorías configuradas:",
                len(CATEGORIAS)
            )

            return []

        indices = np.argsort(
            probabilidades
        )[::-1][:3]

        resultados = []

        for indice in indices:

            resultados.append(
                (
                    CATEGORIAS[indice],
                    float(
                        probabilidades[indice] * 100
                    )
                )
            )

        return resultados

    except Exception as e:

        st.error(
            "❌ Error durante la predicción."
        )

        st.code(str(e))

        return []


def reiniciar_dibujo():

    st.session_state.canvas_version += 1
    st.session_state.terminado = False


def nueva_partida():

    st.session_state.categoria = None
    st.session_state.canvas_version += 1
    st.session_state.terminado = False


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown("""
<div class="header">

<h1>🎨 P R O Y E C T O 1</h1>

<p style="font-size:29px;font-weight:800;">
Adivina Mi Dibujo
</p>

<p>
Dibuja un objeto y deja que la Inteligencia Artificial
intente reconocerlo.
</p>

</div>
""", unsafe_allow_html=True)

# ============================================================
# CATEGORÍAS ARRIBA
# ============================================================

st.markdown(
    '<div class="section-title">🎯 ¿Qué quieres dibujar?</div>',
    unsafe_allow_html=True
)

cols = st.columns(5)

for i, categoria in enumerate(CATEGORIAS):

    with cols[i]:

        if st.button(
            f"{ICONOS[categoria]} {NOMBRES[categoria]}",
            key=f"categoria_{categoria}",
            use_container_width=True
        ):

            st.session_state.categoria = categoria
            st.session_state.canvas_version += 1
            st.session_state.terminado = False

# ============================================================
# ESPERAR SELECCIÓN
# ============================================================

if st.session_state.categoria is None:

    st.info(
        "👆 Selecciona una categoría de arriba "
        "para comenzar el reto."
    )

    st.markdown("""
    <div class="panel">

    <h3 style="text-align:center;">
    🎮 ¿Cómo jugar?
    </h3>

    <p style="text-align:center;font-size:16px;">
    1️⃣ Selecciona qué objeto quieres dibujar.<br>
    2️⃣ Dibuja dentro del lienzo.<br>
    3️⃣ Observa cómo la IA analiza tu dibujo.<br>
    4️⃣ Pulsa <b>Terminar dibujo</b> para comprobar el resultado.
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ============================================================
# CATEGORÍA ACTUAL
# ============================================================

categoria_actual = st.session_state.categoria

st.markdown(
    f"""
    <div class="selected-box">

    <div class="selected-small">
    🎯 TU RETO
    </div>

    <div class="selected-big">
    {ICONOS[categoria_actual]}
    Dibuja un {NOMBRES[categoria_actual]}
    </div>

    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# DOS COLUMNAS
# ============================================================

col1, col2 = st.columns(
    [1.1, 0.9],
    gap="large"
)

# ============================================================
# CANVAS
# ============================================================

with col1:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🖌️ Tu dibujo"
    )

    st.caption(
        "Dibuja con el mouse, dedo o lápiz óptico."
    )

    canvas_result = st_canvas(

        fill_color="black",

        stroke_width=10,

        stroke_color="white",

        background_color="black",

        width=500,

        height=500,

        drawing_mode="freedraw",

        display_toolbar=True,

        key=f"canvas_{st.session_state.canvas_version}"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # BOTONES

    b1, b2 = st.columns(2)

    with b1:

        if st.button(
            "🧹 Borrar dibujo",
            use_container_width=True
        ):

            reiniciar_dibujo()
            st.rerun()

    with b2:

        if st.button(
            "🔄 Nuevo reto",
            use_container_width=True
        ):

            nueva_partida()
            st.rerun()

# ============================================================
# PREDICCIÓN
# ============================================================

with col2:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🤖 Inteligencia Artificial"
    )

    if canvas_result.image_data is not None:

        imagen = np.array(
            canvas_result.image_data
        )

        imagen_rgb = imagen[:, :, :3]

        imagen_gris = np.mean(
            imagen_rgb,
            axis=2
        )

        # Cantidad de píxeles dibujados
        cantidad_tinta = np.sum(
            imagen_gris > 20
        )

        if cantidad_tinta < 80:

            st.info(
                "✏️ Comienza a dibujar para "
                "activar la predicción."
            )

        else:

            resultados = predecir_top3(
                imagen_gris
            )

            if resultados:

                principal = resultados[0]

                nombre_principal = principal[0]
                confianza = principal[1]

                # =================================================
                # SI LA CONFIANZA ES SUFICIENTE
                # =================================================

                if confianza >= UMBRAL_CONFIANZA:

                    st.markdown(
                        f"""
                        <div class="prediction-main">

                        <div class="prediction-label">
                        MI PREDICCIÓN
                        </div>

                        <div class="prediction-name">
                        {ICONOS[nombre_principal]}
                        {NOMBRES[nombre_principal]}
                        </div>

                        <div class="prediction-confidence">
                        Confianza:
                        <b>{confianza:.2f}%</b>
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # =================================================
                # SI LA CONFIANZA ES BAJA
                # =================================================

                else:

                    st.markdown(
                        f"""
                        <div class="prediction-main">

                        <div class="prediction-label">
                        🤔 RESULTADO
                        </div>

                        <div class="prediction-name">
                        ❓ No reconozco este dibujo
                        </div>

                        <div class="prediction-confidence">
                        Mayor confianza:
                        <b>{NOMBRES[nombre_principal]}
                        {confianza:.2f}%</b>
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "La confianza es demasiado baja. "
                        "Intenta dibujar claramente una de "
                        "las categorías disponibles."
                    )

                # =================================================
                # TOP 3
                # =================================================

                st.markdown(
                    "#### 📊 Top 3 predicciones"
                )

                medallas = [
                    "🥇",
                    "🥈",
                    "🥉"
                ]

                for i, (
                    nombre,
                    porcentaje
                ) in enumerate(resultados):

                    st.markdown(
                        f"""
                        <div class="top-card">

                        <span class="top-name">
                        {medallas[i]}
                        {ICONOS[nombre]}
                        {NOMBRES[nombre]}
                        </span>

                        <span class="top-percent">
                        {porcentaje:.2f}%
                        </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            porcentaje / 100,
                            1.0
                        )
                    )

                # =================================================
                # BOTÓN TERMINAR
                # =================================================

                if st.button(
                    "✅ Terminar dibujo",
                    use_container_width=True
                ):

                    st.session_state.terminado = True
                    st.rerun()

                # =================================================
                # RESULTADO FINAL
                # =================================================

                if st.session_state.terminado:

                    st.divider()

                    if confianza < UMBRAL_CONFIANZA:

                        st.warning(
                            "🤔 La IA no tiene suficiente "
                            "confianza para reconocer el dibujo."
                        )

                    elif nombre_principal == categoria_actual:

                        st.success(
                            f"🎉 ¡ACIERTO! La IA reconoció "
                            f"{NOMBRES[nombre_principal]}."
                        )

                    else:

                        st.error(
                            f"❌ La IA predijo "
                            f"{NOMBRES[nombre_principal]}, "
                            f"pero el reto era "
                            f"{NOMBRES[categoria_actual]}."
                        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CATEGORÍAS DISPONIBLES
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="panel">',
    unsafe_allow_html=True
)

st.markdown(
    "### 🎯 Categorías disponibles"
)

for categoria in CATEGORIAS:

    st.markdown(
        f"""
        <span style="
        display:inline-block;
        background:#eef2ff;
        color:#4f46e5;
        padding:9px 15px;
        border-radius:20px;
        margin:4px;
        font-weight:700;
        ">
        {ICONOS[categoria]}
        {NOMBRES[categoria]}
        </span>
        """,
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "🧠 Arquitectura",
        "CNN"
    )

with info2:

    st.metric(
        "🎯 Categorías",
        "5"
    )

with info3:

    st.metric(
        "📊 Predicciones",
        "Top 3"
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🎨 <b>Proyecto 1 — Adivina Mi Dibujo</b>

<br><br>

Clasificación de dibujos mediante
Redes Neuronales Convolucionales.

<br>

🍎 Apple &nbsp; • &nbsp;
🍌 Banana &nbsp; • &nbsp;
🐶 Dog &nbsp; • &nbsp;
🐱 Cat &nbsp; • &nbsp;
🚗 Car

</div>
""", unsafe_allow_html=True)

