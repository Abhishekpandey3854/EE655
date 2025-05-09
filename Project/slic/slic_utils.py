import numpy as np
import tensorflow as tf
from skimage.segmentation import slic
from skimage.color import label2rgb

def generate_slic_mean_color_np(image):
    image_np = image.numpy()
    image_np = np.clip(image_np, 0.0, 1.0)
    segments = slic(image_np, n_segments=150, compactness=10.0, sigma=1.0,
                    start_label=1, channel_axis=-1, enforce_connectivity=True, convert2lab=True)
    mean_color = label2rgb(segments, image=image_np, kind='avg', bg_label=0)
    return mean_color.astype(np.float32)

def generate_slic_target_tf(image_tensor):
    target = tf.py_function(generate_slic_mean_color_np, [image_tensor], tf.float32)
    target.set_shape((128, 128, 3))
    return target
