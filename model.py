import classificatorlib
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


preprocessor = classificatorlib.HyperspectralPreprocessor(image_shape=None)
dataset=preprocessor.load_images_with_labels("./3_hand_out", "./5_hand_out")

print(f"Numero totale di immagini nel dataset: {len(dataset)}")
print(f"Dimensione prima immagine: {dataset[0][0].shape}")
print(f"Label immagine: {dataset[100][1]}")

random.shuffle(dataset)
train_ratio = 0.75
validation_ratio = 0.15
test_ratio = 0.10

# Estrai immagini (X) e label (y) dal dataset
X = np.array([item[0] for item in dataset])  # Tutte le immagini
y = np.array([item[1] for item in dataset])  # Tutte le label

# Dividi il dataset in training, validation e test set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(1 - train_ratio), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(test_ratio / (test_ratio + validation_ratio)), random_state=42)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

X_train = X_train.astype('float16')
X_val = X_val.astype('float16')
X_test = X_test.astype('float16')



# Verifica le lunghezze
print(f"Lunghezza X_train: {len(X_train)}, y_train: {len(y_train)}")
print(f"Lunghezza X_val: {len(X_val)}, y_val: {len(y_val)}")
print(f"Lunghezza X_test: {len(X_test)}, y_test: {len(y_test)}")
print(X_train.shape)
plt.imshow(X_train[12][:, :, 12, :], cmap='gray')
plt.show()

params={
    'layers':3,
    'nodes':256, 
    'kernel': 3
}

classifier = classificatorlib.classificatorModel(params)

classifier.build_cnn((215, 407, 24, 1))

classifier.fit_evaluation(X_train, y_train, X_test, y_test, X_val, y_val, 1, 4)