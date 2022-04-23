import tensorflow_hub as hub
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import cv2
from PIL import Image


# Load in model from tensorflow hub
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


# Clean and preprocess image
def load_image(img_path):
    im = Image.open(img_path).convert('RGB')
    im.save(f'images/original/{img_path.split(".")[0]}.jpg', 'JPEG')
    img_path = f'{img_path.split(".")[0]}.jpg'
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

# List of images to stylize
images = ['images/original/luiz-diaz.webp', 'images/original/salah.webp', 'images/original/taa.webp', 'images/original/thiago.webp', 'images/original/van-dijk.webp', 'images/original/lp.webp']

# Loop through images and stylize each using the model
for image in images:
    content_image = load_image(image)
    style_image = load_image('images/style/style.webp')

    stylized_image = model(tf.constant(content_image), tf.constant(style_image))

    cv2.imwrite(f'images/original/generated/stylized_{image.split(".")[0]}.jpg', cv2.cvtColor(np.squeeze(stylized_image) * 255, cv2.COLOR_RGB2BGR))
