import tensorflow as tf

# Specifica il percorso del modello salvato
saved_model_path = "./models/saved_model"  # Modello SavedModel
# saved_model_path = "path_to_your_model/model.h5"  # Modello .h5 (se è un file .h5)

# Percorso di output per il file TFLite
tflite_model_path = "./models/apple_model_rasp.tflite"

# Carica il modello (SavedModel o .h5)
if saved_model_path.endswith(".h5"):
    model = tf.keras.models.load_model(saved_model_path)
else:
    model = tf.saved_model.load(saved_model_path)

# Configura il convertitore TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model) if saved_model_path.endswith(".h5") \
    else tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

# Abilita TF Select Ops
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Operazioni native di TFLite
    tf.lite.OpsSet.SELECT_TF_OPS     # Operazioni di fallback TF Select
]

# (Opzionale) Ottimizzazione del modello
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Converti il modello
tflite_model = converter.convert()

# Salva il file TFLite
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Modello convertito e salvato come {tflite_model_path}")
