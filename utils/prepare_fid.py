import random
import jax
import jax.numpy as jnp
import tensorflow as tf
from tqdm import tqdm
from fid import get_fid_network
import numpy as np
import tempfile
import os


def load_and_preprocess_image(example):
    try:
        # Decode the raw image bytes instead of parsing as tensor
        image = tf.io.decode_image(example["image"], channels=3)

        # Add shape validation
        if image.shape.ndims != 3 or image.shape[-1] != 3:
            print(f"Invalid image shape: {image.shape}")
            return tf.zeros([299, 299, 3], dtype=tf.float32)

        # Ensure we're working with RGB images
        image = tf.ensure_shape(image, (None, None, 3))

        # Center square crop
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        min_side = tf.minimum(height, width)

        # Calculate crop coordinates
        y_offset = (height - min_side) // 2
        x_offset = (width - min_side) // 2

        # Perform the crop
        image = tf.image.crop_to_bounding_box(
            image, y_offset, x_offset, min_side, min_side
        )

        # Resize to 299x299 for InceptionV3
        image = tf.image.resize(image, [299, 299], method="bilinear")
        # Scale to [-1, 1]
        image = (tf.cast(image, tf.float32) / 127.5) - 1.0
        return image
    except tf.errors.InvalidArgumentError as e:
        print(f"Error decoding image: {e}")
        return tf.zeros([299, 299, 3], dtype=tf.float32)


def calculate_target_stats(
    dataset_path, batch_size=64, max_samples=None, output_path="data/imagenet_stats.npz"
):
    print("Loading validation dataset...")
    val_pattern = f"{dataset_path}/val/images_*.tfrecord"
    val_files = tf.io.gfile.glob(val_pattern)
    random.shuffle(val_files)
    print(f"Found {len(val_files)} validation files")

    dataset = tf.data.TFRecordDataset(val_files, num_parallel_reads=16)
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "synset": tf.io.FixedLenFeature([], tf.string),
        "class_name": tf.io.FixedLenFeature([], tf.string),
    }

    def parse_tfrecord(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    dataset = dataset.map(parse_tfrecord, num_parallel_calls=16)
    dataset = dataset.map(lambda x: load_and_preprocess_image(x), num_parallel_calls=16)
    dataset = dataset.filter(lambda x: x is not None)  # Remove failed images
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size, drop_remainder=True, num_parallel_calls=16)

    print("Getting Inception network...")
    inception_fn = get_fid_network()

    print("Calculating activation statistics...")
    all_activations = []
    total_samples = 0

    for batch in tqdm(dataset):
        batch_np = jnp.array(batch.numpy())
        batch_np = batch_np.reshape((jax.device_count(), -1) + batch_np.shape[1:])
        activations = inception_fn(batch_np)
        all_activations.append(jnp.reshape(activations, (-1, activations.shape[-1])))

        total_samples += batch.shape[0]
        if max_samples and total_samples >= max_samples:
            break

    all_activations = jnp.concatenate(all_activations, axis=0)

    if max_samples:
        all_activations = all_activations[:max_samples]

    mu = jnp.mean(all_activations, axis=0)
    sigma = jnp.cov(all_activations, rowvar=False)

    print(f"Calculated stats from {len(all_activations)} images")
    print(f"Mean shape: {mu.shape}, Covariance shape: {sigma.shape}")

    stats_data = {"mu": np.array(mu), "sigma": np.array(sigma)}

    if output_path.startswith("gs://"):
        local_path = "data/imagenet_stats.npz"
        np.savez(local_path, **stats_data)
        tf.io.gfile.copy(local_path, output_path, overwrite=True)
        print(f"✅ Copied statistics to GCS: {output_path}")
    else:
        np.savez(output_path, **stats_data)
        print(f"✅ Saved target statistics to {output_path}")

    return mu, sigma


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="gs://diffdata/imagenet")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples to use for statistics calculation",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="gs://diffdata/imagenet_stats.npz",
        help="Path to save the statistics (can be local or gs:// bucket path)",
    )
    args = parser.parse_args()

    calculate_target_stats(
        args.dataset_path, args.batch_size, args.max_samples, args.output_path
    )
