import tensorflow as tf
import tensorflow_datasets as tfds
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

def load_dataset(batch_size=16):
    train_ds = tfds.load("tf_flowers", split="train[:80%]", shuffle_files=True)
    train_ds = train_ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds


def main():
    train_ds = load_dataset()
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="mae")

    model.fit(train_ds, epochs=30, callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True)
    ])

if __name__ == "__main__":
    main()
