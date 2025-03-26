# Clasificador de Perros y Gatos con Flask y TensorFlow Lite

Este proyecto es una aplicación web hecha con **Flask** que utiliza modelos de **Deep Learning** para clasificar en tiempo real si una imagen de la cámara corresponde a un **perro**, un **gato**, o si no es claro.

Permite seleccionar entre dos modelos:
- Una red **convolucional (CNN)**.
- Una red **fully connected (FC)**.

## Características
- Acceso a la cámara del dispositivo desde el navegador (compatible con móvil y escritorio).
- Predicción en tiempo real usando imagenes capturadas.
- Clasificación en tres estados: `Perro`, `Gato`, `No claro`.
- Selector de modelo entre CNN y FC.
- Backend optimizado con modelos en formato **.tflite** (TensorFlow Lite) para despliegue liviano.

---

## Estructura del proyecto

```
mi_app/
├── app.py                  # Servidor Flask principal
├── requirements.txt        # Dependencias para Render o local
├── Procfile                # Instrucciones para Render
├── model/
│   ├── cnn_model.tflite    # Modelo CNN optimizado
│   └── fc_model.tflite     # Modelo FC optimizado
├── static/
│   └── main.js             # Lógica JS de captura y predicción
├── templates/
│   └── index.html          # Interfaz HTML con Bootstrap
└── README.md               # Este archivo
```

---

## Requisitos

- Python 3.8+
- TensorFlow 2.x
- Flask
- gunicorn (para despliegue)

Instalar dependencias:
```bash
pip install -r requirements.txt
```

---

## Ejecución local

```bash
python app.py
```

Luego abre tu navegador en:
```
http://127.0.0.1:5000
```

---

## Despliegue en Render

1. Sube este proyecto a un repositorio en GitHub.
2. Crea una cuenta en [https://render.com](https://render.com)
3. En Render:
   - Selecciona "New Web Service"
   - Conecta tu repositorio
   - Configura:
     - **Build command**: `pip install -r requirements.txt`
     - **Start command**: `gunicorn app:app`
4. Asegúrate de tener `cnn_model.tflite` y `fc_model.tflite` dentro de `model/`, y que pesen <100MB.

---

## Licencia
Este proyecto es de uso libre con fines educativos y experimentales.

---

## Autor
**Tu Nombre**  
[GitHub](https://github.com/AngelBReal)  
[LinkedIn](https://www.linkedin.com/in/angelbarrazareal/)  

---

¡Listo para usar! Si necesitas ayuda para desplegarlo o extenderlo, contáctame.

