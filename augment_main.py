import classificatorlib
import matplotlib.pyplot as plt
import numpy as np

input_folder = "./3_hand_in"  # Cartella contenente immagini concatenate
output_folder = "./3_hand_out"  # Cartella per immagini preprocessate
file_type = "hdr"  # Tipo di file: "hdr" o "raw"
num_bands = 25  # Numero di bande spettrali

# Configura il preprocessore
preprocessor = classificatorlib.HyperspectralPreprocessor(image_shape=None)  # L'immagine sarà dinamica

# Carica immagini dal formato specifico
if file_type == "hdr":
    images = preprocessor.load_hdr_images_from_folder(input_folder)
elif file_type == "raw":
    images = preprocessor.load_raw_images_from_folder(input_folder)
else:
    raise ValueError(f"file format unsupported: {file_type}")

    # Applica il preprocessing a tutte le immagini
processed_images = preprocessor.preprocess_all_images(images, num_bands)

    # Salva immagini preprocessate
preprocessor.save_images_to_folder(processed_images, output_folder, file_extension="npy")

print(processed_images.shape)
print(f"Preprocessing ultimated. Images saved in: {output_folder}")

plt.imshow(processed_images[1, :, :, 4])
plt.show()
