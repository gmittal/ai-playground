import functools
import itertools
import pathlib

import chex
import colorama
import flax.linen as nn
import jax
import jax.numpy as jnp
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


class CoordConv2D(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple
    with_r: bool

    @nn.compact
    def __call__(self, x):
        batch_size, dim_y, dim_x, _ = x.shape
        xx_ones = jnp.ones((1, 1, 1, dim_x), dtype=jnp.int32)
        yy_ones = jnp.ones((1, 1, 1, dim_y), dtype=jnp.int32)

        xx_range = jnp.arange(dim_y, dtype=jnp.int32)
        yy_range = jnp.arange(dim_x, dtype=jnp.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = xx_range @ xx_ones
        yy_channel = yy_range @ yy_ones

        # transpose y
        yy_channel = jnp.transpose(yy_channel, (0, 1, 3, 2))

        xx_channel = xx_channel.astype(jnp.float32) / (dim_y - 1)
        yy_channel = yy_channel.astype(jnp.float32) / (dim_x - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # convert to (b, h, w, c)
        xx_channel = jnp.transpose(xx_channel, (0, 2, 3, 1))
        yy_channel = jnp.transpose(yy_channel, (0, 2, 3, 1))

        xx_channel = xx_channel.repeat(batch_size, axis=0)
        yy_channel = yy_channel.repeat(batch_size, axis=0)

        x_with_coords = jnp.concatenate((x, xx_channel, yy_channel), axis=-1)

        if self.with_r:
            r_channel = jnp.sqrt(
                jnp.power(xx_channel - 0.5, 2) + jnp.power(yy_channel - 0.5, 2)
            )
            x_with_coords = jnp.concatenate((x_with_coords, r_channel), axis=-1)

        x = nn.Conv(self.features, self.kernel_size, self.strides)(x_with_coords)
        return x


class Net(nn.Module):
    @nn.compact
    def __call__(self, x):

        # dump mlp, probably will need fourier feats to improve
        # for the high frequency jump for the square
        # x = nn.Dense(512)(x)
        # x = nn.relu(x)
        # x = nn.Dense(64 * 64)(x)
        # x = x.reshape(x.shape[0], 64, 64)
        # return x

        # dumb deconv
        # c = 1
        # fs = 2
        # x = nn.ConvTranspose(64 * c, (fs, fs), (1, 2))(x[:, :, None, None])
        # x = nn.relu(x)
        # x = nn.ConvTranspose(64 * c, (fs, fs), (2, 2))(x)
        # x = nn.relu(x)
        # x = nn.ConvTranspose(64 * c, (fs, fs), (2, 2))(x)
        # x = nn.relu(x)
        # x = nn.ConvTranspose(32 * c, (fs, fs), (2, 2))(x)
        # x = nn.relu(x)
        # x = nn.ConvTranspose(32 * c, (fs, fs), (2, 2))(x)
        # x = nn.relu(x)
        # x = nn.ConvTranspose(1 * c, (fs, fs), (2, 2))(x)
        # return x[:, :, :, 0]

        # coordconv, works!
        c = 1
        fs = 1
        x = jnp.tile(x[:, None, None, :], (1, 64, 64, 1))
        x = CoordConv2D(32, (1, 1), (1, 1), with_r=True)(x)
        x = nn.Conv(64, (1, 1), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(64, (1, 1), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(1, (1, 1), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(1, (1, 1), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(8 * c, (fs, fs), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(16 * c, (fs, fs), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(16 * c, (fs, fs), (1, 1))(x)
        x = nn.relu(x)
        x = nn.Conv(1, (fs, fs), (1, 1))(x)
        return x[:, :, :, 0]


class NotSoCleverDataset(Dataset):
    def __init__(self):
        self.onehots = np.pad(
            np.eye(3136).reshape((3136, 56, 56, 1)),
            ((0, 0), (4, 4), (4, 4), (0, 0)),
            "constant",
        )
        conv2d = jax.vmap(
            functools.partial(jax.scipy.signal.convolve2d, mode='same'),
            in_axes=(0, None),
        )
        self.images = np.array(conv2d(self.onehots[:, :, :, 0], np.ones((9, 9))))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        one_idx = np.where(self.onehots[idx])
        x, y = one_idx[0][0], one_idx[1][0]
        return {'x': np.array([x, y]) / 64.0, 'y': self.images[idx]}


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
    model = Net()
    params = model.init(rng, jnp.ones([1, 2]))['params']

    # TODO: make optimizer configurable
    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    coords = batch['x']
    output = batch['y']

    def loss_fn(params):
        pred_out = state.apply_fn({'params': params}, coords)
        chex.assert_equal_shape([pred_out, output])
        loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits=pred_out, labels=output)
        )
        return loss, pred_out

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits)


@functools.partial(
    jax.jit,
    static_argnums=(
        2,
        3,
    ),
)
def generate_image(state, x, y):
    recon = state.apply_fn({"params": state.params}, jnp.array([x, y])[None, :] / 64.0)
    recon = nn.sigmoid(recon) > 0.5
    return jnp.uint8(recon * 255)[0]


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config
    np.random.seed(config.np_seed)

    # setup data
    train_dataset = NotSoCleverDataset()
    dataloader = DataLoader(
        train_dataset,
        collate_fn=numpy_collate,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    data_iter = itertools.cycle(dataloader)
    examples_seen = 0

    # setup model and state
    ckpt_dir = pathlib.Path(FLAGS.workdir) / 'checkpoints'
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    x = train_dataset[0]['x'][None, :]
    tabulate_fn = nn.tabulate(Net(), tabulate_rng)
    logging.info(tabulate_fn(x))

    # train
    while state.step < config.train_steps:
        batch = next(data_iter)
        state, (loss, logits) = train_step(state, batch)

        # compute additional metrics
        train_pred = jnp.bool_(nn.sigmoid(logits) > 0.5)
        train_real = jnp.bool_(batch['y'])
        chex.assert_equal_shape([train_pred, train_real])
        accuracy = jnp.mean(train_pred == train_real)
        intersection = (train_pred & train_real).sum(axis=(1, 2))
        union = (train_pred | train_real).sum(axis=(1, 2))
        jaccard = jnp.mean(intersection / union)

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | loss {loss.item():.4f} '
                f'| accuracy {accuracy.item():.4f} | iou {jaccard.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            recon = generate_image(state, 32, 32)
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
