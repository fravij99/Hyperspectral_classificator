# Hyperspectral Image Processing Library

![Python](https://img.shields.io/badge/python-3.8%2B-blue)  
![MIT License](https://img.shields.io/badge/license-MIT-green)

<table>
<tr>
    <td align="center">
        <img src="https://raw.githubusercontent.com/fravij99/Hyperspectral_classificator/master/images/hyperspectral_agricolture.png" width="600" alt="Original Foam Image">
        <p><strong>Original Foam Image</strong></p>
    </td>
<table>

A Python library for processing, augmenting, and classifying hyperspectral and RAW images. This library provides tools for preprocessing hyperspectral data, implementing data augmentation techniques, and building a 3D Convolutional Neural Network (CNN) for binary classification tasks.

## Features

- **Preprocessing Hyperspectral and RAW Images:**
  - Reshape 2D concatenated hyperspectral images into 3D format.
  - Load and process `.hdr` and `.raw` image files.
  - Save processed images in `.npy` or `.tiff` formats.

- **Data Augmentation:**
  - Horizontal and vertical flips.
  - 90° multiple rotations.
  - Gaussian noise addition.
  - Random cropping of images.

- **3D Convolutional Neural Network (CNN):**
  - Build customizable 3D CNN architectures for binary classification tasks.
  - Model training, evaluation, and visualization of training history.
  - Generate confusion matrix for performance evaluation.
  - Save and export trained models in TensorFlow's SavedModel or TensorFlow Lite format.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hyperspectral-library.git
   cd hyperspectral-library

## Dependencies
The library relies on the following Python packages:

- `numpy`
- `random`
- `scipy`
- `scikit-image`
- `spectral`
- `rawpy`
- `os`
- `tifffile`
- `tqdm`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `plot-keras-history`

Make sure all dependencies are installed before running the library.

## Usage

### Preprocessing Functions

- **reshape_to_3d(self, concatenated_image, num_bands):** 
  Transforms a 2D concatenated hyperspectral image into a 3D image by splitting it into `num_bands` along the width.

- **flip_horizontal(self, image_3d):** Flips the given 3D image horizontally.

- **flip_vertical(self, image_3d):**  Flips the given 3D image vertically.

- **rotate(self, image_3d, angle):**    Rotates the 3D image by a specified angle (must be a multiple of 90°).

- **add_noise(self, image_3d, mode='gaussian', var=0.01):**    Adds random noise to the 3D image. By default, it uses Gaussian noise with a variance of 0.01.

- **random_crop(self, image_3d, crop_size):**    Extracts a random crop of size `crop_size` (height, width) from the input 3D image.

- **augment(self, image_3d):**    Applies a set of data augmentation techniques (e.g., flips, rotations, noise) to the input 3D image and returns a list of augmented images.

- **preprocess_all_images(self, images, num_bands):**    Applies preprocessing and data augmentation to a list of input images. Returns a list of processed and augmented images.

- **load_hdr_images_from_folder(self, folder_path):**    Loads `.hdr` hyperspectral images from the specified folder and converts them into NumPy arrays.

- **load_raw_images_from_folder(self, folder_path):**    Loads `.raw` images from the specified folder, processes them using `rawpy`, and converts them into NumPy arrays.

- **save_images_to_folder(self, images, folder_path, file_extension):**   Saves a list of 3D images to the specified folder in the chosen file format (`.npy` or `.tiff`).

- **load_images_with_labels(self, folder_three_fingers, folder_five_fingers):**    Loads `.npy` images from two folders, assigns labels (`0` for "three fingers", `1` for "five fingers"), and returns a labeled dataset as a list of `(image, label)` tuples.

---

### CNN Model Functions

- **build_cnn(self, input_shape):**    Builds a 3D Convolutional Neural Network (CNN) for binary classification. Allows customization of network depth, filter size, kernel size, and dense layer parameters.

- **fit_evaluation(self, Xtrain, Ytrain, Xtest, Ytest, Xval, Yval, epochs, batch_size):**    Trains the CNN using the provided training data, evaluates it on the test set, and generates a confusion matrix for performance analysis.

- **saving_model(self):**    Saves the trained CNN model in TensorFlow's `SavedModel` format.

- **convert_model(self, path):**   Converts the trained TensorFlow model into TensorFlow Lite format for deployment on edge devices. Supports both TFLite and TensorFlow Select operations.

## Examples
You can find example scripts in the repository for common tasks, such as preprocessing and training a 3D CNN.

## Results 

| **Training** | **Confusion Matrix** | **Model Parameters** | 
|-------------------|----------------------------------------------|-------------------------------|
![Training](https://raw.githubusercontent.com/fravij99/Hyperspectral_classificator/master/images/training_apple.png) | ![Confusion matrix](https://raw.githubusercontent.com/fravij99/Hyperspectral_classificator/master/images/confusion_apple.png) | ![Model Patameters](https://raw.githubusercontent.com/fravij99/Hyperspectral_classificator/master/images/conv3d_params.png) | 

|**Classified as good** | **Classified as rotten** |
|-----------------------------------|-------------------------------------------------------------|
|![Fractal Fit](https://raw.githubusercontent.com/fravij99/Hyperspectral_classificator/master/images/good_apple_test.png) | ![Scaling Fit](https://raw.githubusercontent.com/fravij99/Hyperspectral_classificator/master/images/apple_rotten_test.png) |

---

## License
This project is licensed under the MIT License.

