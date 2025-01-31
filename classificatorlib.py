import numpy as np
import random
from scipy.ndimage import rotate
from skimage.util import random_noise
from spectral import open_image  # Per leggere immagini HDR
import rawpy  # Per leggere immagini RAW
import os
import tifffile
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import resize
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from plot_keras_history import plot_history
from tensorflow.keras.layers import Input


class HyperspectralPreprocessor:
    def __init__(self, image_shape):
        """
        image_shape: Tuple (heigth, width, band_num), 
        """
        self.image_shape = image_shape
        self.target_shape=(53, 100, 25)

    def reshape_to_3d(self, concatenated_image, num_bands):
        """
        Turns the 2D image in a 3D image with bands as depth
        """
        print(concatenated_image.shape)
        height, width_bands = concatenated_image.shape
        band_width = width_bands // num_bands  # Calcola la larghezza di ogni banda
        
        # Lista per raccogliere le bande ritagliate
        bands = []
        for i in range(num_bands):
            start_col = i * band_width
            end_col = (i + 1) * band_width
            band = concatenated_image[:, start_col:end_col]  # Ritaglia la banda
            bands.append(band)

        # Converte la lista di bande in un array 3D (height, band_width, num_bands)
        image_3d = np.stack(bands, axis=-1)
        return image_3d
    

    def flip_horizontal(self, image_3d):
        return np.flip(image_3d, axis=1)

    def flip_vertical(self, image_3d):
        return np.flip(image_3d, axis=0)

    def rotate(self, image_3d, angle):
        """
        Rotate image: angle is a 90° multiple
        """
        return rotate(image_3d, angle, axes=(0, 1), reshape=False, mode='nearest')

    def random_rotate(self, image_3d):
        """
        Rotate image by a random angle (not limited to 90° multiples).
        """
        random_angle = random.uniform(0, 360)  # Angolo casuale tra 0° e 360°
        return rotate(image_3d, random_angle, axes=(0, 1), reshape=False, mode='nearest')

    def add_noise(self, image_3d, mode='gaussian'):
        return random_noise(image_3d, mode=mode, clip=False)

    def random_crop(self, image_3d, crop_size):
        """
        Crop a random region of the image.
        """
        height, width, _ = image_3d.shape
        crop_h, crop_w = crop_size
        start_h = random.randint(0, height - crop_h)
        start_w = random.randint(0, width - crop_w)
        return image_3d[start_h:start_h + crop_h, start_w:start_w + crop_w, :]



    def random_brightness_adjustment(self, image_3d, num_bands=10, brightness_factor_range=(0.5, 2.0)):
        adjusted_image = image_3d.copy()
        total_bands = image_3d.shape[2]

        # Seleziona bande casuali
        selected_bands = random.sample(range(total_bands), num_bands)
        brightness_factor = random.uniform(*brightness_factor_range)

        for band in selected_bands:
            # Incrementa la luminosità senza clipping
            adjusted_image[:, :, band] = adjusted_image[:, :, band] * brightness_factor


        return adjusted_image



    def augment(self, image_3d):
        augmented_images = []
        
        # Flip
        augmented_images.append(self.random_brightness_adjustment(self.flip_horizontal(image_3d)))
        augmented_images.append(self.random_brightness_adjustment(self.flip_vertical(image_3d)))
        
        # Rotations
        for angle in range(90, -91, -10):
            augmented_images.append(self.random_brightness_adjustment(self.rotate(image_3d, angle)))
        
        # Random rotation
        augmented_images.append(self.random_brightness_adjustment(self.random_rotate(image_3d)))
        
        # Add noise
        augmented_images.append(self.random_brightness_adjustment(self.add_noise(image_3d)))
        
        # Brightness adjustment
        augmented_images.append(self.random_brightness_adjustment(image_3d))
        
        
        return augmented_images
    

    def preprocess_all_images(self, images, num_bands):
        """
        Apply preprocessing to all the concatenated images given 
        """
        processed_images = []
        for idx, image in tqdm(enumerate(images), total=len(images), desc="Processing images"):
            
            # Applica la data augmentation
            augmented_images = self.augment(image)
            
            # Aggiungi tutte le immagini augmentate come elementi separati
            processed_images.extend(augmented_images)
        
        return np.array(processed_images)
    
    
    def resize_image(self, image, target_shape):
        if target_shape is not None:
            return resize(image, target_shape, anti_aliasing=True, preserve_range=True)
        return image
    

    def load_hdr_images_from_folder(self, folder_path):
        images = []
        for file_name in tqdm(os.listdir(folder_path), desc='Loading HDR images...'):
            if file_name.endswith(".hdr"):
                file_path = os.path.join(folder_path, file_name)
                hdr_image = open_image(file_path).load()  # Carica l'immagine HDR
                hdr_image = np.array(hdr_image)  # Converte in numpy array
                resized_image = self.resize_image(hdr_image, self.target_shape)
                images.append(resized_image)
        return images


    def load_raw_images_from_folder(self, folder_path):
        images = []
        for file_name in tqdm(os.listdir(folder_path), desc='Loading RAW images...'):
            if file_name.endswith(".raw"):
                file_path = os.path.join(folder_path, file_name)
                with rawpy.imread(file_path) as raw:
                    raw_image = raw.postprocess()
                    resized_image = self.resize_image(raw_image, self.target_shape)
                    images.append(resized_image)
        return images
    

    def save_images_to_folder(self, images, folder_path, file_extension):
        """
        Salva immagini preprocessate in una cartella.
        
        :param images: Lista di immagini 3D.
        :param folder_path: Percorso della cartella di destinazione.
        :param file_extension: Estensione del file da salvare (default: "npy").
        """
        os.makedirs(folder_path, exist_ok=True)
        for i, image in enumerate(images):
            file_name = f"processed_image_{i}.{file_extension}"
            file_path = os.path.join(folder_path, file_name)
            if file_extension == "npy":
                np.save(file_path, image)
            elif file_extension in ["tiff", "tif"]:
                
                tifffile.imwrite(file_path, image.astype(np.float32))

    
    def process_folder(self, dataset, folder_path, label):
            for file_name in tqdm(os.listdir(folder_path), desc='Loading images...'):
                if file_name.endswith(".npy"):  # Controlla il formato dei file .npy
                    file_path = os.path.join(folder_path, file_name)
                    npy_image = np.load(file_path)  # Carica l'immagine .npy
                    dataset.append((npy_image, label))  # Aggiungi una tupla di immagine e label

    def test_model_folder(self, dataset, folder_path):
            for file_name in tqdm(os.listdir(folder_path), desc='Loading images...'):
                if file_name.endswith(".npy"):  # Controlla il formato dei file .npy
                    file_path = os.path.join(folder_path, file_name)
                    npy_image = np.load(file_path)  # Carica l'immagine .npy
                    dataset.append((npy_image))  # Aggiungi una tupla di immagine e label


    def load_images_with_labels(self, folder_three_fingers, folder_five_fingers):
        dataset = []
        
        

        # Processa le cartelle
        self.process_folder(dataset, folder_three_fingers, label=0)  # Tre dita
        self.process_folder(dataset, folder_five_fingers, label=1)  # Cinque dita

        return dataset
    
    def plot_all_bands(self, image_3d, cmap='gray', title="Bande Spettrali"):
        num_bands = image_3d.shape[2]  # Numero di bande spettrali
        cols = 5  # Numero di colonne per la griglia
        rows = (num_bands + cols - 1) // cols  # Calcola il numero di righe necessario

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        fig.suptitle(title, fontsize=16)

        # Trova i valori globali min e max dell'immagine per mantenerli fissi
        vmin, vmax = image_3d.min(), image_3d.max()

        # Visualizza ogni banda nella griglia
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols]  # Calcola la posizione nella griglia
            if i < num_bands:
                ax.imshow(image_3d[:, :, i], cmap=cmap, vmin=vmin, vmax=vmax)  # Blocca il range
                ax.set_title(f"Banda {i+1}")
                ax.axis('off')
            else:
                ax.axis('off')  # Nascondi subplot vuoti

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Spazio per il titolo
        plt.show()


class classificatorModel:
    def __init__(self, params):
        self.params = params
    
    def build_cnn(self, input_shape):
        """
        Crea un modello CNN 3D avanzato per la classificazione video (binaria).

        Args:
        - input_shape (tuple): Shape dei frame di input (ad esempio, (16, 112, 112, 3) per 16 frame di dimensione 112x112 e 3 canali).
        
        Returns:
        - model (Sequential): Modello CNN 3D compilato.
        """
        model = Sequential()
        model.add(Input(shape=input_shape))  # Usa Input al posto di input_shape in Conv3D
        for i in range(self.params['layers']):
            model.add(Conv3D(self.params['nodes'], (self.params['kernel'], self.params['kernel'], self.params['kernel']), activation='relu', padding='same'))
            model.add(MaxPooling3D((2, 2, 2)))
            model.add(BatchNormalization())

        # Flatten layer to flatten the output of the convolutional layers
        model.add(Flatten())
        # Fully connected (dense) layer with 512 units and ReLU activation
        model.add(Dense(self.params['dense'], activation='relu'))
        # Dropout layer with dropout rate of 0.5
        model.add(Dropout(0.5))
        # Output layer with one unit for binary classification (sigmoid activation)
        model.add(Dense(1, activation='sigmoid'))

        # Compile the model with Adam optimizer and binary crossentropy loss for binary classification
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

        return model
    
    def fit_evaluation(self, Xtrain, Ytrain, Xtest, Ytest, Xval, Yval, epochs, batch_size):
        # Se Ytrain e Ytest sono one-hot encoded, convertili in formato scalare (se necessario)
        Ytrain = np.argmax(Ytrain, axis=1) if Ytrain.ndim > 1 else Ytrain
        Ytest = np.argmax(Ytest, axis=1) if Ytest.ndim > 1 else Ytest
        Yval = np.argmax(Yval, axis=1) if Yval.ndim > 1 else Yval
        
        hist = self.model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval), epochs=epochs, batch_size=batch_size)
        plot_history(hist)
        
        loss, accuracy = self.model.evaluate(Xtest, Ytest)
        
        y_pred = self.model.predict(Xtest)
        
        # Predizioni probabilistiche di sigmoid, quindi le classi sono 0 o 1
        y_pred_classes = (y_pred > 0.5).astype(int)  # soglia di 0.5 per classificazione binaria
        conf_matrix = confusion_matrix(Ytest, y_pred_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()


    def saving_model(self):
        # Salva il modello nel formato SavedModel
        self.model.export("/content/drive/MyDrive/grape_classificator/saved_model")
        print("Modello salvato in formato SavedModel (/content/drive/MyDrive/grape_classificator/saved_model).")

    
    def convert_model_savedmodel(self, saved_model_path, tflite_model_path, quantize=False):

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

        # Abilita TF Select Ops per Conv3D e MaxPool3D
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # Abilita la quantizzazione dinamica, se richiesto
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Converte il modello
        try:
            tflite_model = converter.convert()
            # Salva il modello TFLite
            with open(tflite_model_path, "wb") as f:
                f.write(tflite_model)
            print(f"Modello TFLite salvato con successo in: {tflite_model_path}")
        except Exception as e:
            print(f"Errore durante la conversione: {e}")







