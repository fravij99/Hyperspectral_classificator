import numpy as np
import matplotlib.pyplot as plt
from tflite_runtime.interpreter import Interpreter  # Usa tflite_runtime per il modello
import os
from tqdm import tqdm  # Per la barra di progresso

# Funzione per caricare i dati non etichettati
def test_model_folder(dataset, folder_path):
    for file_name in tqdm(os.listdir(folder_path), desc='Loading images...'):
        if file_name.endswith(".npy"):  # Controlla il formato dei file .npy
            file_path = os.path.join(folder_path, file_name)
            npy_image = np.load(file_path)  # Carica l'immagine .npy
            dataset.append(npy_image)  # Aggiungi l'immagine al dataset

# Caricamento dei dati
dataset = []
test_model_folder(dataset, "./apple_trial_out")
dataset = np.array(dataset)  # Converte in un array NumPy
dataset = np.expand_dims(dataset, axis=-1)  # Aggiunge il canale
dataset = dataset.astype('float32')  # Assicura il tipo corretto

print(f"Forma del dataset: {dataset.shape}")

# Carica il modello TFLite e alloca i tensori
interpreter = Interpreter(model_path="models/apple_model_rasp.tflite")
interpreter.allocate_tensors()

# Verifica la forma richiesta dal modello
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"Forma richiesta dal modello: {input_details[0]['shape']}")

# Prepara l'immagine di test
test_image = dataset[4:5]  # Seleziona un'immagine (aggiunge automaticamente la dimensione del batch)
print(f"Forma dell'immagine di test: {test_image.shape}")

# Esegui inferenza
interpreter.set_tensor(input_details[0]['index'], test_image)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])[0]

# Interpreta la predizione
prediction = (output > 0.5).astype(int)
print(f"Predizione grezza: {output}")
print(f"Classe predetta: {prediction}")

# Mostra l'immagine e il risultato
plt.imshow(dataset[4, :, :, 2, 0])  # Mostra il secondo canale
if prediction == 1:
    plt.title("Apple is good")
else:
    plt.title("Apple is rotten")
plt.show()
