import numpy as np
import random
from scipy.ndimage import rotate
from skimage.util import random_noise



class HyperspectralPreprocessor:
    def __init__(self, image_shape):
        """
        image_shape: Tuple (heigth, width, band_num), 
        """
        self.image_shape = image_shape

    def reshape_to_3d(self, concatenated_image, num_bands):
        """
        Turns the 2D image in a 3D image with bands as depth
        """
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

    def add_noise(self, image_3d, mode='gaussian', var=0.01):
        """
        Add noise to the image
        :param mode:  noise type (default: 'gaussian').
        :param var: noise variance (only for 'gaussian').
        """
        return random_noise(image_3d, mode=mode, var=var)

    def random_crop(self, image_3d, crop_size):
    
        height, width, _ = image_3d.shape
        crop_h, crop_w = crop_size
        start_h = random.randint(0, height - crop_h)
        start_w = random.randint(0, width - crop_w)
        return image_3d[start_h:start_h + crop_h, start_w:start_w + crop_w, :]

    def augment(self, image_3d):

        augmented_images = []
        
        # Flip
        augmented_images.append(self.flip_horizontal(image_3d))
        augmented_images.append(self.flip_vertical(image_3d))
        
        # Rotations
        for angle in [90, 180, 270]:
            augmented_images.append(self.rotate(image_3d, angle))
        
        # Add noise
        augmented_images.append(self.add_noise(image_3d))
        
        # Crop
        crop_size = (int(self.image_shape[0] * 0.8), int(self.image_shape[1] * 0.8))
        augmented_images.append(self.random_crop(image_3d, crop_size))
        
        return augmented_images
    

    def preprocess_all_images(self, images, num_bands):
        """
        Apply preprocessing to all the concatenated images given 
        """
        processed_images = []
        for idx, image in enumerate(images):
            image_3d = self.reshape_to_3d(image, num_bands)
            processed_images.append(image_3d)
        return np.array(processed_images)