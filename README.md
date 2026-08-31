


# 🎨 Proyecto 1 — Adivina Mi Dibujo

## 📌 Descripción

Adivina Mi Dibujo es una aplicación desarrollada con Python, TensorFlow/Keras y Streamlit que permite al usuario realizar dibujos mediante un lienzo interactivo y obtener predicciones utilizando un modelo de Redes Neuronales Convolucionales (CNN).

La aplicación realiza las predicciones mientras el usuario dibuja y muestra las tres categorías más probables junto con su porcentaje de confianza.

---

## 🎯 Objetivo

Desarrollar un sistema capaz de clasificar dibujos realizados con mouse, dedo o lápiz óptico en tiempo real.

El sistema trabaja con cinco categorías:

- 🍎 Apple
- 🍌 Banana
- 🐶 Dog
- 🐱 Cat
- 🚗 Car

---

## 🧠 Modelo

El sistema utiliza un modelo de clasificación basado en una Red Neuronal Convolucional (CNN), desarrollado utilizando TensorFlow/Keras.

El modelo entrenado se encuentra en:

`modelo_adivina_dibujo.h5`

---

## 📚 Dataset

Para el entrenamiento se utilizó un subconjunto del dataset Quick, Draw! de Google.

Las categorías utilizadas en el proyecto son:

- Apple
- Banana
- Dog
- Cat
- Car

Cada categoría fue preparada para el entrenamiento y evaluación del modelo.

---

## 💻 Tecnologías utilizadas

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- Streamlit Drawable Canvas

---

## 🖌️ Funcionamiento de la aplicación

1. El usuario selecciona una categoría.
2. La aplicación muestra el reto que debe dibujar.
3. El usuario realiza el dibujo sobre el lienzo.
4. El sistema procesa la imagen.
5. La imagen se convierte a escala de grises.
6. Se redimensiona a 28 × 28 píxeles.
7. El modelo CNN realiza la predicción.
8. Se muestran las tres predicciones más probables.
9. Se muestra el porcentaje de confianza.
10. La aplicación proporciona retroalimentación visual sobre el resultado.

---

## 🔮 Predicciones

La aplicación muestra:

### 🥇 Primera predicción
La categoría con mayor probabilidad.

### 🥈 Segunda predicción
La segunda categoría más probable.

### 🥉 Tercera predicción
La tercera categoría más probable.

Cada resultado incluye su porcentaje de confianza.

---

## ⚠️ Predicciones con baja confianza

El sistema utiliza un umbral de confianza para evitar afirmar que un dibujo pertenece a una categoría cuando el modelo no está suficientemente seguro.

Si ninguna predicción supera el nivel de confianza establecido, la aplicación muestra un mensaje indicando que no reconoce claramente el dibujo.

Esto permite manejar dibujos que no pertenecen a las cinco categorías utilizadas durante el entrenamiento.

---

## 📊 Evaluación del modelo

Accuracy de validación:

Pérdida en prueba:

**0.3345**

Accuracy de prueba:

**89.90%**

Matriz de confusión:

**<img width="649" height="547" alt="image" src="https://github.com/user-attachments/assets/2453a92a-c8b6-4d7c-a355-1424e9f45764" />**


---

## ▶️ Instalación

Primero instalar las dependencias:

```bash
pip install -r requirements.txt
````

Después ejecutar la aplicación:

```bash
streamlit run app.py
```

La aplicación se abrirá en el navegador.

---

## 📁 Estructura del proyecto

```text
AdivinaMiDibujo/
│
├── app.py
├── modelo_adivina_dibujo.h5
├── requirements.txt
└── README.md
```
## 📓 Notebook de Google Colab

El código utilizado para la preparación del dataset, entrenamiento,
evaluación y generación del modelo se encuentra disponible en Google Colab.

👉 https://colab.research.google.com/drive/14kg75i8Q4ndS-dnk7NvwfJs3yeosTq1O#scrollTo=3uLzU-J7qkoo&uniqifier=2
---

👉 Vercel-Despliegue

**https://vercel.com/pri-57e2/proyecto-final-adivina-mi-dibujo**


👉 ** Video de defensa**

https://ister-my.sharepoint.com/:f:/g/personal/kelly_leyton_ister_edu_ec/IgATEL-Z7JzmTqQC-YYmQ90qAe1P956jJXLJQIPOcyCvu0Q?e=jLp12N&xsdata=%3D&sdata=SFhBTmpQUHhLQ2JCMnMyc3l2YnMyZUtvUlJJRVI3QjNDalcwVlEwV2VlZz0%3D&ovuser=e9763399-4de0-4078-a8dc-a6cb985b4841%2Cpriscila.quinonez%40ister.edu.ec


## 🎥 Demostración

Durante la demostración se muestran diferentes ejemplos de dibujos realizados en tiempo real.

La aplicación permite observar:

* Predicción en vivo.
* Top 3 de categorías.
* Porcentaje de confianza.
* Retroalimentación visual.
* Aciertos.
* Errores o predicciones con baja confianza.

También se muestra un caso en el que el modelo puede presentar dificultades al recibir un dibujo que no pertenece a las categorías utilizadas durante el entrenamiento.

---

## 👩‍💻 Proyecto académico

**Proyecto 1 — Adivina Mi Dibujo**

Aplicación de clasificación de dibujos mediante Inteligencia Artificial y Redes Neuronales Convolucionales.

