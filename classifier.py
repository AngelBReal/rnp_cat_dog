# classifiers.py
# Entrenamiento de clasificadores de perros y gatos usando TensorFlow y Keras en entorno SSH con SLURM

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para entornos sin GUI (ej: SSH)
import sys
import os
import contextlib

# Context manager para suprimir la salida en consola (evita logs de tfds y otros)
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

if __name__ == "__main__":

    # Aseguramos que se use la URL correcta del dataset
    setattr(tfds.image_classification.cats_vs_dogs, '_URL',
            "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

    # Descargar dataset de cats vs dogs y dividir en entrenamiento y prueba
    with suppress_output():
        (datos_entrenamiento, datos_prueba), metadatos = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:]'],
            as_supervised=True,
            with_info=True
        )

    TAMANO_IMG = 100  # Tamaño al que se redimensionan las imágenes

    # Preprocesamiento: redimensionar y normalizar
    def procesar_imagen(imagen, etiqueta):
        imagen = tf.image.resize(imagen, [TAMANO_IMG, TAMANO_IMG])
        imagen = tf.cast(imagen, tf.float32) / 255.0
        return imagen.numpy(), etiqueta.numpy()

    X, y = [], []

    # Procesar datos de entrenamiento
    print("Procesando imágenes de entrenamiento...")
    for imagen, etiqueta in tqdm(datos_entrenamiento):
        img, lbl = procesar_imagen(imagen, etiqueta)
        X.append(img)
        y.append(lbl)

    # Procesar datos de prueba
    print("Procesando imágenes de prueba...")
    for imagen, etiqueta in tqdm(datos_prueba):
        img, lbl = procesar_imagen(imagen, etiqueta)
        X.append(img)
        y.append(lbl)

    # Convertir a arrays de NumPy
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)

    # Dividir en conjunto de entrenamiento y validación
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Callbacks para entrenamiento
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0)
    checkpoint_fc = ModelCheckpoint(
        '/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/modelo/best_fc.keras',
        monitor='val_accuracy', save_best_only=True, verbose=0)
    callbacks_list_fc = [early_stop, checkpoint_fc]

    # Modelo fully connected (base)
    modelo_fc = Sequential([
        Flatten(input_shape=(TAMANO_IMG, TAMANO_IMG, 3)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    modelo_fc.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\nEntrenando modelo Fully Connected...")
    history_fc = modelo_fc.fit(
        X_train, y_train,
        epochs=10,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list_fc,
        verbose=0
    )

    # Generador de datos con aumentos para CNN
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Función para construir modelos con Hiperparámetros para KerasTuner
    def build_model(hp):
        model = Sequential()
        model.add(Conv2D(hp.Choice('conv1_filters', [32, 64]), (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(hp.Float('dropout1', 0.2, 0.5, step=0.1)))

        model.add(Conv2D(hp.Choice('conv2_filters', [64, 128]), (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(hp.Float('dropout2', 0.2, 0.5, step=0.1)))

        model.add(Flatten())
        model.add(Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'))
        model.add(Dropout(hp.Float('dropout3', 0.2, 0.5, step=0.1)))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Buscar mejores hiperparámetros con Random Search
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/tuner',
        project_name='cnn_aug_dropout',
        overwrite=True
    )

    tuner.search(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=10,
        callbacks=[early_stop],
        verbose=0
    )

    # Obtener el mejor modelo según los resultados del tuner
    best_model = tuner.get_best_models(1)[0]

    # Entrenar modelo CNN con mejores hiperparámetros
    history_best = best_model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=5,
        callbacks=[early_stop],
        verbose=0
    )

    print("\nEvaluación modelo Fully Connected:")
    modelo_fc.evaluate(X_test, y_test, verbose=0)

    print("\nEvaluación modelo CNN mejorado:")
    best_model.evaluate(X_test, y_test, verbose=0)

    # Graficar historia de entrenamiento y guardar resultados
    def plot_history(history, title):
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Validación')
        plt.title(f'Precisión - {title}')
        plt.xlabel('Epocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.savefig(f'/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/plots/{title.lower().replace(" ", "_")}_accuracy.png')
        plt.close()

    plot_history(history_fc, "Fully Connected")
    plot_history(history_best, "CNN Mejorado")

    # Guardar ambos modelos en formatos .h5 y .keras
    modelo_fc.save('/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/modelo/h5/perros-gatos-fc-ad.h5')
    modelo_fc.save('/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/modelo/keras/perros-gatos-fc-ad.keras')

    best_model.save('/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/modelo/h5/cnn_tuned_aug_dropout.h5')
    best_model.save('/LUSTRE/home/rn_lcc_01/scripts/RNP2025/cat_dog/modelo/keras/cnn_tuned_aug_dropout.keras')