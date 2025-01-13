from typing import Any
from absl import app, flags
from functools import partial
import random
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import wandb

import jax
import jax.numpy as jnp
import flax
import optax
from ml_collections import config_flags
import ml_collections
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

from psgd_jax.kron import kron
from psgd_jax import precond_update_prob_schedule
from soap_jax.soap import scale_by_soap

from utils.wandb import setup_wandb, default_wandb_config
from utils.train_state import TrainState, target_update
from utils.checkpoint import Checkpoint
from utils.stable_vae import StableVAE
from diffusion_transformer import DiT
from utils.classes import IMAGENET2012_CLASSES

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset_name",
    "imagenet256",
    "Environment name. Can be imagenet256 or imagenet128.",
)
flags.DEFINE_string("load_dir", None, "Logging dir (if not None, save params).")
flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")
flags.DEFINE_string(
    "fid_stats", "gs://diffdata/imagenet_stats.npz", "FID stats file. Set to None to disable FID calculation."
)
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("log_interval", 200, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("save_interval", 200000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1_000_000), "Number of training steps.")

flags.DEFINE_string(
    "gs_data_path", "gs://diffdata/imagenet", "Base GCS path for TFRecord files."
)
flags.DEFINE_integer("fid_samples", 10000, "Number of samples to use for FID calculation.")
model_config = ml_collections.ConfigDict(
    {
        # Make sure to run with Large configs when we actually want to run!
        "optimizer": "adam",  # "kron", "soap", or "adam"
        "lr": 0.0001,
        "warmup_steps": 1000,
        "beta1": 0.9,
        "beta2": 0.99,
        "weight_decay": 0.0001,
        "hidden_size": 64,  # set by preset
        "patch_size": 8,  # set by preset
        "depth": 2,  # set by preset
        "num_heads": 2,  # set by preset
        "mlp_ratio": 1,  # set by preset
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "denoise_timesteps": 32,
        "cfg_scale": 4.0,
        "target_update_rate": 1.0,  # 0.9999
        "t_sampler": "log-normal",
        "t_conditioning": 1,
        "preset": "debug",
        "use_stable_vae": 1,
    }
)

preset_configs = {
    "debug": {
        "hidden_size": 64,
        "patch_size": 8,
        "depth": 2,
        "num_heads": 2,
        "mlp_ratio": 1,
    },
    "big": {
        "hidden_size": 768,
        "patch_size": 2,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4,
    },
    "big_thin": {
        "hidden_size": 512,
        "patch_size": 2,
        "depth": 30,
        "num_heads": 8,
        "mlp_ratio": 3,
    },
    "medium": {
        "hidden_size": 768,
        "patch_size": 2,
        "depth": 16,
        "num_heads": 12,
        "mlp_ratio": 3,
    },
    "semilarge": {  # local-batch of 32 achieved, (16 with eps)
        "hidden_size": 1024,
        "patch_size": 2,
        "depth": 22,  # Should be 24, but this fits in memory better.
        "num_heads": 16,
        "mlp_ratio": 4,
    },
    "large": {  # local-batch of 2 achieved
        "hidden_size": 1024,
        "patch_size": 2,
        "depth": 24,
        "num_heads": 16,
        "mlp_ratio": 4,
    },
    "xlarge": {
        "hidden_size": 1152,
        "patch_size": 2,
        "depth": 28,
        "num_heads": 16,
        "mlp_ratio": 4,
    },
}

wandb_config = default_wandb_config()
wandb_config.update({"project": "flow", "name": "flow_{dataset_name}"})

config_flags.DEFINE_config_dict("wandb", wandb_config, lock_config=False)
config_flags.DEFINE_config_dict("model", model_config, lock_config=False)

##############################################
## Model Definitions.
##############################################


# x_0 = Noise
# x_1 = Data
def get_x_t(images, eps, t):
    x_0 = eps
    x_1 = images
    t = jnp.clip(t, 0, 1 - 0.01)  # Always include a little bit of noise.
    return (1 - t) * x_0 + t * x_1


def get_v(images, eps):
    x_0 = eps
    x_1 = images
    return x_1 - x_0


class FlowTrainer(flax.struct.PyTreeNode):
    rng: Any
    model: TrainState
    model_eps: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    # Train
    @partial(jax.pmap, axis_name="data")
    def update(self, images, labels, pmap_axis="data"):
        new_rng, label_key, time_key, noise_key = jax.random.split(self.rng, 4)

        def loss_fn(params):
            # Sample a t for training.
            if self.config["t_sampler"] == "log-normal":
                t = jax.random.normal(time_key, (images.shape[0],))
                t = 1 / (1 + jnp.exp(-t))
            elif self.config["t_sampler"] == "uniform":
                t = jax.random.uniform(time_key, (images.shape[0],), minval=0, maxval=1)

            t_full = t[:, None, None, None]  # [batch, 1, 1, 1]
            eps = jax.random.normal(noise_key, images.shape)
            x_t = get_x_t(images, eps, t_full)
            v_t = get_v(images, eps)

            if self.config["t_conditioning"] == 0:
                t = jnp.zeros_like(t)

            v_prime = self.model(
                x_t,
                t,
                labels,
                train=True,
                rngs={"label_dropout": label_key},
                params=params,
            )
            loss = jnp.mean((v_prime - v_t) ** 2)

            return loss, {
                "l2_loss": loss,
                "v_abs_mean": jnp.abs(v_t).mean(),
                "v_pred_abs_mean": jnp.abs(v_prime).mean(),
            }

        grads, info = jax.grad(loss_fn, has_aux=True)(self.model.params)
        grads = jax.lax.pmean(grads, axis_name=pmap_axis)
        info = jax.lax.pmean(info, axis_name=pmap_axis)

        updates, new_opt_state = self.model.tx.update(
            grads, self.model.opt_state, self.model.params
        )
        new_params = optax.apply_updates(self.model.params, updates)
        new_model = self.model.replace(
            step=self.model.step + 1, params=new_params, opt_state=new_opt_state
        )

        info["grad_norm"] = optax.global_norm(grads)
        info["update_norm"] = optax.global_norm(updates)
        info["param_norm"] = optax.global_norm(new_params)

        # Update the model_eps
        if self.config["target_update_rate"] == 1:
            new_model_eps = new_model
        else:
            new_model_eps = target_update(
                self.model, self.model_eps, self.config["target_update_rate"]
            )
        new_trainer = self.replace(
            rng=new_rng, model=new_model, model_eps=new_model_eps
        )
        return new_trainer, info

    @partial(jax.jit, static_argnames=("cfg"))
    def call_model(self, images, t, labels, cfg=True, cfg_val=1.0):
        if self.config["t_conditioning"] == 0:
            t = jnp.zeros_like(t)
        if not cfg:
            return self.model_eps(images, t, labels, train=False, force_drop_ids=False)
        else:
            labels_uncond = (
                jnp.ones(labels.shape, dtype=jnp.int32) * self.config["num_classes"]
            )  # Null token
            images_expanded = jnp.tile(images, (2, 1, 1, 1))  # (batch*2, h, w, c)
            t_expanded = jnp.tile(t, (2,))  # (batch*2,)
            labels_full = jnp.concatenate([labels, labels_uncond], axis=0)
            v_pred = self.model_eps(
                images_expanded,
                t_expanded,
                labels_full,
                train=False,
                force_drop_ids=False,
            )
            v_label = v_pred[: images.shape[0]]
            v_uncond = v_pred[images.shape[0] :]
            v = v_uncond + cfg_val * (v_label - v_uncond)
            return v

    @partial(
        jax.pmap,
        axis_name="data",
        in_axes=(0, 0, 0, 0),
        static_broadcasted_argnums=(4, 5),
    )
    def call_model_pmap(self, images, t, labels, cfg=True, cfg_val=1.0):
        return self.call_model(images, t, labels, cfg=cfg, cfg_val=cfg_val)


##############################################
## Training Code.
##############################################
def main(_):
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    preset_dict = preset_configs[FLAGS.model.preset]
    for k, v in preset_dict.items():
        FLAGS.model[k] = v

    np.random.seed(FLAGS.seed)
    print("Using devices", jax.local_devices())
    device_count = len(jax.local_devices())
    global_device_count = jax.device_count()
    print("Device count", device_count)
    print("Global device count", global_device_count)
    local_batch_size = FLAGS.batch_size // (global_device_count // device_count)
    print("Global Batch: ", FLAGS.batch_size)
    print("Node Batch: ", local_batch_size)
    print("Device Batch:", local_batch_size // device_count)

    # Create wandb logger
    if jax.process_index() == 0:
        setup_wandb(FLAGS.model.to_dict(), **FLAGS.wandb)

    def get_dataset(is_train):
        """Load dataset from TFRecords.

        Args:
            is_train (bool): Whether to load training or validation split.
        """
        feature_description = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "synset": tf.io.FixedLenFeature([], tf.string),
            "class_name": tf.io.FixedLenFeature([], tf.string),
        }

        def parse_tfrecord(example_proto):
            parsed = tf.io.parse_single_example(example_proto, feature_description)
            image = tf.io.decode_image(parsed["image"], channels=3)
            image = tf.cast(image, tf.float32)

            height = tf.shape(image)[0]
            width = tf.shape(image)[1]
            min_dim = tf.minimum(height, width)
            if is_train:
                image = tf.image.random_crop(image, size=[min_dim, min_dim, 3])
                image = tf.image.random_flip_left_right(image)
            else:
                height_offset = (height - min_dim) // 2
                width_offset = (width - min_dim) // 2
                image = tf.image.crop_to_bounding_box(
                    image, height_offset, width_offset, min_dim, min_dim
                )

            img_size = 256 if "256" in FLAGS.dataset_name else 128
            if img_size != image.shape[0]:
                image = tf.image.resize(image, (img_size, img_size), antialias=True)
            image = image / 127.5 - 1.0
            return image, parsed["label"]

        split = "train" if is_train else "val"
        file_pattern = f"{FLAGS.gs_data_path}/{split}/images*.tfrecord"
        filenames = tf.io.gfile.glob(file_pattern)
        if not filenames:
            raise ValueError(f"No TFRecord files found at {file_pattern}")
        filenames = filenames[jax.process_index()::jax.process_count()]
        print(f"Process {jax.process_index()}: Using {len(filenames)} files from {split} split")
        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=16 if is_train else 4
        )
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()
        dataset = dataset.batch(
            local_batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        )

        def prepare_batch(images, labels):
            images = tf.reshape(
                images, (device_count, -1, images.shape[1], images.shape[2], 3)
            )
            labels = tf.reshape(labels, (device_count, -1))
            return images, labels

        dataset = dataset.map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)

        options = tf.data.Options()
        options.deterministic = False
        dataset = dataset.with_options(options)

        dataset = dataset.prefetch(32).as_numpy_iterator()
        return iter(dataset)

    dataset = get_dataset(is_train=True)
    dataset_valid = get_dataset(is_train=False)
    example_obs, example_labels = next(dataset)
    example_obs = example_obs[0, :1]

    if FLAGS.model.use_stable_vae:
        vae = StableVAE.create()
        example_obs = vae.encode(jax.random.PRNGKey(0), example_obs)
        vae_rng = flax.jax_utils.replicate(jax.random.PRNGKey(42))
        vae_encode_pmap = jax.pmap(vae.encode)
        vae_decode = jax.jit(vae.decode)
        vae_decode_pmap = jax.pmap(vae.decode)

    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, param_key, dropout_key = jax.random.split(rng, 3)
    print(
        "Total Memory on device:",
        float(jax.local_devices()[0].memory_stats()["bytes_limit"]) / 1024**3,
        "GB",
    )

    ###################################
    # Creating Model and put on devices.
    ###################################
    FLAGS.model.image_channels = example_obs.shape[-1]
    FLAGS.model.image_size = example_obs.shape[1]
    dit_args = {
        "patch_size": FLAGS.model["patch_size"],
        "hidden_size": FLAGS.model["hidden_size"],
        "depth": FLAGS.model["depth"],
        "num_heads": FLAGS.model["num_heads"],
        "mlp_ratio": FLAGS.model["mlp_ratio"],
        "class_dropout_prob": FLAGS.model["class_dropout_prob"],
        "num_classes": FLAGS.model["num_classes"],
    }
    model_def = DiT(**dit_args)

    example_t = jnp.zeros((1,))
    example_label = jnp.zeros((1,), dtype=jnp.int32)
    model_rngs = {"params": param_key, "label_dropout": dropout_key}
    params = model_def.init(model_rngs, example_obs, example_t, example_label)["params"]
    print(
        "Total num of parameters:",
        sum(x.size for x in jax.tree_util.tree_leaves(params)),
    )

    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0, FLAGS.model["lr"], FLAGS.model["warmup_steps"]),
            optax.constant_schedule(FLAGS.model["lr"]),
        ],
        boundaries=[FLAGS.model["warmup_steps"]]
    )
    if FLAGS.model["optimizer"] == "adam":
        tx = optax.adamw(
            learning_rate=lr_schedule,
            b1=FLAGS.model["beta1"],
            b2=FLAGS.model["beta2"],
            weight_decay=FLAGS.model["weight_decay"],
        )
    elif FLAGS.model["optimizer"] == "kron":
        # which layers are scanned
        all_false = jax.tree.map(lambda _: False, params)
        scanned_layers = flax.traverse_util.ModelParamTraversal(
            lambda p, _: "scan" in p or "Scan" in p
        ).update(lambda _: True, all_false)

        tx = kron(
            learning_rate=lr_schedule,
            weight_decay=FLAGS.model["weight_decay"],
            preconditioner_update_probability=precond_update_prob_schedule(
                flat_start=1000, min_prob=0.05
            ),
            scanned_layers=scanned_layers,
        )
    elif FLAGS.model["optimizer"] == "soap":
        tx = optax.chain(
            scale_by_soap(precondition_frequency=20),
            # exploding gradients so clipping updates like kron instead of grad clipping
            optax.clip_by_block_rms(1.1),
            optax.add_decayed_weights(FLAGS.model["weight_decay"]),
            optax.scale_by_learning_rate(lr_schedule),
        )
    else:
        raise ValueError(f"Unknown optimizer: {FLAGS.model['optimizer']}")

    model_ts = TrainState.create(model_def, params, tx=tx)
    model_ts_eps = TrainState.create(model_def, params)
    model = FlowTrainer(rng, model_ts, model_ts_eps, FLAGS.model)

    if FLAGS.load_dir is not None:
        cp = Checkpoint(FLAGS.load_dir)
        model = cp.load_model(model)
        print("Loaded model with step", model.model.step)
        del cp

    if FLAGS.fid_stats is not None:
        from utils.fid import get_fid_network, fid_from_stats

        get_fid_activations = get_fid_network()
        if FLAGS.fid_stats.startswith("gs://"):
            with tf.io.gfile.GFile(FLAGS.fid_stats, 'rb') as f:
                truth_fid_stats = np.load(f)
        else:
            truth_fid_stats = np.load(FLAGS.fid_stats)

    model = flax.jax_utils.replicate(model, devices=jax.local_devices())
    model = model.replace(rng=jax.random.split(rng, len(jax.local_devices())))
    jax.debug.visualize_array_sharding(
        model.model.params["FinalLayer_0"]["Dense_0"]["bias"]
    )

    valid_images_small, valid_labels_small = next(dataset_valid)
    valid_images_small = valid_images_small[:, :1]
    valid_labels_small = valid_labels_small[:, :1]
    visualize_labels = example_labels[:, :1]
    imagenet_labels = list(IMAGENET2012_CLASSES.values())
    if FLAGS.model.use_stable_vae:
        valid_images_small = vae_encode_pmap(vae_rng, valid_images_small)

    ###################################
    # Train Loop
    ###################################

    def eval_model():
        # Needs to be in a separate function so garbage collection works correctly.

        # Validation Losses
        valid_images, valid_labels = next(dataset_valid)
        if FLAGS.model.use_stable_vae:
            valid_images = vae_encode_pmap(vae_rng, valid_images)
        _, valid_update_info = model.update(valid_images, valid_labels)
        valid_update_info = jax.tree.map(lambda x: x.mean(), valid_update_info)
        valid_metrics = {f"validation/{k}": v for k, v in valid_update_info.items()}
        if jax.process_index() == 0:
            wandb.log(valid_metrics, step=i)

        def process_img(img):
            if FLAGS.model.use_stable_vae:
                img = vae_decode(img[None])[0]
            img = img * 0.5 + 0.5
            img = jnp.clip(img, 0, 1)
            img = np.array(img)
            return img

        # Training loss on various t.
        mse_total = []
        for t in np.arange(0, 11):
            key = jax.random.PRNGKey(42)
            t = t / 10
            t_full = jnp.full((batch_images.shape), t)
            t_vector = jnp.full((batch_images.shape[0], batch_images.shape[1]), t)
            eps = jax.random.normal(key, batch_images.shape)
            x_t = get_x_t(batch_images, eps, t_full)
            v = get_v(batch_images, eps)
            pred_v = model.call_model_pmap(x_t, t_vector, batch_labels, False, 0.0)
            assert pred_v.shape == v.shape
            mse_loss = jnp.mean((v - pred_v) ** 2)
            mse_total.append(mse_loss)
            if jax.process_index() == 0:
                wandb.log({f"training_loss_t/{t}": mse_loss}, step=i)
        mse_total = jnp.array(mse_total[1:-1])
        if jax.process_index() == 0:
            wandb.log({"training_loss_t/mean": mse_total.mean()}, step=i)

        # Validation loss on various t.
        mse_total = []
        fig, axs = plt.subplots(3, 10, figsize=(30, 20))
        for t in np.arange(0, 11):
            key = jax.random.PRNGKey(42)
            t = t / 10
            t_full = jnp.full((valid_images.shape), t)
            t_vector = jnp.full((valid_images.shape[0], valid_images.shape[1]), t)
            eps = jax.random.normal(key, valid_images.shape)
            x_t = get_x_t(valid_images, eps, t_full)
            v = get_v(valid_images, eps)
            pred_v = model.call_model_pmap(x_t, t_vector, valid_labels, False, 0.0)
            assert pred_v.shape == v.shape
            mse_loss = jnp.mean((v - pred_v) ** 2)
            mse_total.append(mse_loss)
            if jax.process_index() == 0:
                wandb.log({f"validation_loss_t/{t}": mse_loss}, step=i)
        mse_total = jnp.array(mse_total[1:-1])
        if jax.process_index() == 0:
            wandb.log({"validation_loss_t/mean": mse_total.mean()}, step=i)
            plt.close(fig)

        # One-step denoising at various noise levels.
        assert valid_images.shape[0] == len(jax.local_devices()) # [devices, batch//devices, etc..]
        t = jnp.arange(device_count) / device_count # between 0 and 0.875
        t = jnp.repeat(t[:, None], valid_images.shape[1], axis=1) # [8, batch//devices, etc..] DEVICES=8
        eps = jax.random.normal(key, valid_images.shape)
        x_t = get_x_t(valid_images, eps, t[..., None, None, None])
        v_pred = model.call_model_pmap(x_t, t, valid_labels, False, 0.0)
        x_1_pred = x_t + v_pred * (1-t[..., None, None, None])
        if jax.process_index() == 0:
            # plot comparison witah matplotlib. put each reconstruction side by side.
            fig, axs = plt.subplots(device_count, device_count*3, figsize=(90, 30))
            for j in range(device_count):
                for k in range(device_count):
                    axs[j,3*k].imshow(process_img(valid_images[j,k]), vmin=0, vmax=1)
                    axs[j,3*k+1].imshow(process_img(x_t[j,k]), vmin=0, vmax=1)
                    axs[j,3*k+2].imshow(process_img(x_1_pred[j,k]), vmin=0, vmax=1)
            wandb.log({f'reconstruction_n': wandb.Image(fig)}, step=i)
            plt.close(fig)

        # Full Denoising with CFG=4 only
        key = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps = jax.random.normal(key, valid_images_small.shape)
        delta_t = 1.0 / FLAGS.model.denoise_timesteps

        # Single CFG scale evaluation
        x = eps
        all_x = []
        for ti in range(FLAGS.model.denoise_timesteps):
            t = ti / FLAGS.model.denoise_timesteps
            t_vector = jnp.full((x.shape[0], x.shape[1]), t)
            v = model.call_model_pmap(
                x, t_vector, visualize_labels, True, FLAGS.model.cfg_scale
            )
            x = x + v * delta_t
            if (
                ti % (FLAGS.model.denoise_timesteps // device_count) == 0
                or ti == FLAGS.model.denoise_timesteps - 1
            ):
                all_x.append(np.array(x))
        all_x = np.stack(all_x, axis=2)
        all_x = all_x[:, :, -device_count:]

        if jax.process_index() == 0:
            fig, axs = plt.subplots(device_count, device_count, figsize=(30, 30))
            for j in range(device_count):
                for t in range(device_count):
                    axs[t, j].imshow(process_img(all_x[j, 0, t]), vmin=0, vmax=1)
                axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
            wandb.log({f"sample_cfg_{FLAGS.model.cfg_scale}": wandb.Image(fig)}, step=i)
            plt.close(fig)

        # Single denoise steps evaluation (32 steps only)
        key = jax.random.PRNGKey(42 + jax.process_index() + i)
        eps = jax.random.normal(key, valid_images_small.shape)
        x = eps
        for ti in range(32):  # Fixed at 32 steps
            t = ti / 32
            t_vector = jnp.full((x.shape[0], x.shape[1]), t)
            v = model.call_model_pmap(
                x, t_vector, visualize_labels, True, FLAGS.model.cfg_scale
            )
            x = x + v * (1.0 / 32)

        if jax.process_index() == 0:
            fig, axs = plt.subplots(device_count, device_count, figsize=(30, 30))
            for j in range(device_count):
                for t in range(device_count):
                    axs[t, j].imshow(process_img(x[j, t]), vmin=0, vmax=1)
                axs[0, j].set_title(f"{imagenet_labels[visualize_labels[j, 0]]}")
            wandb.log({f"sample_N/32": wandb.Image(fig)}, step=i)
            plt.close(fig)

        # FID calculation.
        if FLAGS.fid_stats is not None:
            activations = []
            valid_images_shape = valid_images.shape
            num_batches = int(np.ceil(FLAGS.fid_samples / FLAGS.batch_size))

            for fid_it in tqdm.tqdm(
                range(num_batches), 
                desc="Calculating FID", 
                disable=jax.process_index() != 0
            ):
                _, valid_labels = next(dataset_valid)

                key = jax.random.PRNGKey(42 + fid_it)
                x = jax.random.normal(key, valid_images_shape)
                delta_t = 1.0 / FLAGS.model.denoise_timesteps
                for ti in range(FLAGS.model.denoise_timesteps):
                    t = (
                        ti / FLAGS.model.denoise_timesteps
                    )  # From x_0 (noise) to x_1 (data)
                    t_vector = jnp.full((x.shape[0], x.shape[1]), t)
                    if FLAGS.model.cfg_scale == -1:
                        v = model.call_model_pmap(x, t_vector, valid_labels, False, 0.0)
                    else:
                        v = model.call_model_pmap(
                            x, t_vector, valid_labels, True, FLAGS.model.cfg_scale
                        )
                    x = x + v * delta_t
                if FLAGS.model.use_stable_vae:
                    x = vae_decode_pmap(x)
                x = jax.image.resize(
                    x,
                    (x.shape[0], x.shape[1], 299, 299, 3),
                    method="bilinear",
                    antialias=False,
                )
                x = 2 * x - 1
                acts = get_fid_activations(x)[
                    ..., 0, 0, :
                ]  # [devices, batch//devices, 2048]
                acts = jax.pmap(
                    lambda x: jax.lax.all_gather(x, "i", axis=0), axis_name="i"
                )(acts)[
                    0
                ]  # [global_devices, batch//global_devices, 2048]
                acts = np.array(acts)
                activations.append(acts)
            if jax.process_index() == 0:
                activations = np.concatenate(activations, axis=0)
                # Cut off any extra samples beyond what was requested
                activations = activations[:FLAGS.fid_samples]
                activations = activations.reshape((-1, activations.shape[-1]))
                mu1 = np.mean(activations, axis=0)
                sigma1 = np.cov(activations, rowvar=False)
                fid = fid_from_stats(
                    mu1, sigma1, truth_fid_stats["mu"], truth_fid_stats["sigma"]
                )
                wandb.log({
                    "fid": fid,
                }, step=i)

        del valid_images, valid_labels
        del all_x, x, x_t, eps
        print("Finished all the eval stuff")

    for i in tqdm.tqdm(range(FLAGS.max_steps), smoothing=0.1, dynamic_ncols=True):

        batch_images, batch_labels = next(dataset)
        if FLAGS.model.use_stable_vae:
            batch_images = vae_encode_pmap(vae_rng, batch_images)

        model, update_info = model.update(batch_images, batch_labels)

        if i % FLAGS.log_interval == 0:
            update_info = jax.tree.map(lambda x: np.array(x), update_info)
            update_info = jax.tree.map(lambda x: x.mean(), update_info)
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if jax.process_index() == 0:
                wandb.log(train_metrics, step=i)

        if (i % FLAGS.eval_interval == 0 or i == 1000) and i > 0:  # always do one at 1000
            eval_model()

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None and i > 0:
            if jax.process_index() == 0:
                model_single = flax.jax_utils.unreplicate(model)
                cp = Checkpoint(FLAGS.save_dir, parallel=False)
                cp.set_model(model_single)
                cp.save()
                del cp, model_single


if __name__ == "__main__":
    app.run(main)
