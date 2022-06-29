import functools
import itertools
import pathlib

import colorama
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax

from absl import app
from absl import flags
from absl import logging
from flax.training import train_state
from flax.training import checkpoints
from ml_collections import config_flags
from PIL import Image
from torch.utils.data import Dataset, DataLoader

Fore = colorama.Fore
Style = colorama.Style

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x, B):
        variance = self.param('b_var', nn.initializers.ones, (1,))
        B = variance * B
        x_proj = 2.0 * jnp.pi * x @ B.T
        x = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.sigmoid(x)
        return x


class PixelDataset(Dataset):
    def __init__(self, img):
        # img.shape = (H, W, C)
        self.img = img
        self.H, self.W, _ = img.shape
        self.size = self.H * self.W

    def __len__(self):
        return self.size

    def _index_to_coord(self, idx):
        return idx // self.W, idx % self.W

    def get_features(self, idx):
        y, x = self._index_to_coord(idx)
        y = 2.0 * (y / self.H) - 1.0
        x = 2.0 * (x / self.W) - 1.0
        return np.array([y, x])

    def get_label(self, idx):
        return self.img[self._index_to_coord(idx)] / 255.0

    def __getitem__(self, idx):
        return {'x': self.get_features(idx), 'y': self.get_label(idx)}


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return {key: numpy_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        return np.array(batch)


def create_train_state(rng, config):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 2]), jnp.ones((config.kernel_dim, 2)))[
        'params'
    ]

    # TODO: make optimizer configurable
    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch, B):
    coords = batch['x']
    rgb = batch['y']

    def loss_fn(params):
        pred_rgb = state.apply_fn({'params': params}, coords, B)
        chex.assert_equal_shape([pred_rgb, rgb])
        loss = jnp.mean(jnp.sum(jnp.square(pred_rgb - rgb), axis=-1))
        return loss, pred_rgb

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


@functools.partial(
    jax.jit,
    static_argnums=(
        2,
        3,
    ),
)
def generate_image(state, B, height, width):
    x = jnp.arange(width)
    y = jnp.arange(height)
    xx, yy = jnp.meshgrid(x, y, sparse=False)

    yy = 2.0 * (yy / height) - 1.0
    xx = 2.0 * (xx / width) - 1.0
    coords = jnp.stack((yy, xx), axis=2).reshape(-1, 2)

    z_flat = MLP().apply({"params": state.params}, coords, B)
    recon = z_flat.reshape(height, width, 3)
    recon = (recon * 255).astype(np.uint8)
    return recon


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config
    np.random.seed(config.np_seed)

    # setup data
    img = np.array(Image.open(config.image_path))
    train_dataset = PixelDataset(img)
    dataloader = DataLoader(
        train_dataset,
        collate_fn=numpy_collate,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        batch_size=len(train_dataset),
        num_workers=config.num_workers,
    )
    data_iter = itertools.cycle(dataloader)
    examples_seen = 0

    # create Gaussian kernel
    b_kernel = jnp.sqrt(config.initial_variance) * jnp.array(
        np.random.randn(config.kernel_dim, 2)
    )

    # setup model and state
    ckpt_dir = pathlib.Path(FLAGS.workdir) / 'checkpoints'
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    x = train_dataset[0]['x']
    tabulate_fn = nn.tabulate(MLP(), tabulate_rng)
    logging.info(tabulate_fn(x, b_kernel))

    # train
    while state.step < config.train_steps:
        batch = next(data_iter)
        state, loss = train_step(state, batch, b_kernel)

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | loss {loss.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            recon = generate_image(state, b_kernel, train_dataset.H, train_dataset.W)
            img_dir = pathlib.Path(FLAGS.workdir) / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(img_dir / f'{state.step}.png')
            Image.fromarray(np.array(recon)).save(output_path)
            logging.info(
                f'{Fore.GREEN}EVAL:{Style.RESET_ALL} saving reconstruction'
                f' to {output_path}'
            )

        if state.step % config.ckpt_interval == 0:
            checkpoints.save_checkpoint(
                ckpt_dir, state, int(state.step), keep=float('inf')
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    jax.config.config_with_absl()
    app.run(main)
