import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from model.unet import build_unet
from slic.slic_utils import generate_slic_target_tf

def preprocess(image):
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def map_fn(data):
    image = preprocess(data['image'])
    target = generate_slic_target_tf(image)
    return image, target

def load_validation_set(batch_size=4):
    val_ds = tfds.load("tf_flowers", split="train[80%:]", shuffle_files=False)
    val_ds = val_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def show_predictions(model, dataset, num_samples=4):
    for images, targets in dataset.take(1):
        predictions = model.predict(images)
        for i in range(num_samples):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(images[i])
            axs[0].set_title("Input")
            axs[1].imshow(targets[i])
            axs[1].set_title("SLIC Target")
            axs[2].imshow(predictions[i])
            axs[2].set_title("Model Output")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

def main():
    print("Loading model...")
    model = build_unet()
    model.load_weights("best_model.h5")

    print("Loading validation set...")
    val_ds = load_validation_set()

    print("Showing predictions...")
    show_predictions(model, val_ds)

if __name__ == "__main__":
    main()
