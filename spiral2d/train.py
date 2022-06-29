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
    def setup(self):
        self.layers = [
            nn.Dense(256),
            nn.relu,
            nn.Dense(256),
            nn.relu,
            nn.Dense(256),
            nn.relu,
            nn.Dense(1),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ToyDataset(Dataset):
    def __init__(self):
        self.size = 5_000
        rotations = 1.5
        noise_std = 0.05

        self.x, self.y = self._make_spirals(self.size, noise_std, rotations)
        self.y = self.y.reshape(-1, 1)

        self.ds_min = self.x.min()
        self.ds_max = self.x.max()

    def __len__(self):
        return self.size

    def _make_spirals(self, n_samples, noise_std=0.0, rotations=1.0):
        ts = np.linspace(0, 1, n_samples)
        rs = ts**0.5
        thetas = rs * rotations * 2 * np.pi
        signs = np.random.randint(0, 2, (n_samples,)) * 2 - 1
        labels = (signs > 0).astype(int)

        xs = rs * signs * np.cos(thetas) + np.random.randn(n_samples) * noise_std
        ys = rs * signs * np.sin(thetas) + np.random.randn(n_samples) * noise_std
        points = np.stack([xs, ys], axis=1)
        return points, labels

    def _normalize(self, pt):
        return (pt - self.ds_min) / (self.ds_max - self.ds_min)

    def __getitem__(self, idx):
        return {'x': self._normalize(self.x[idx]), 'y': self.y[idx]}


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
    params = model.init(rng, jnp.ones([1, 2]))['params']

    # TODO: make optimizer configurable
    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    images = batch['x']
    labels = batch['y']

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        chex.assert_equal_shape([logits, labels])
        loss = jnp.mean(
            optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits)


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config

    # setup data
    # TODO: add checkpoint-aware recovery for dataloader
    train_dataset = ToyDataset()
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
    x = train_dataset[0]['x']
    tabulate_fn = nn.tabulate(MLP(), tabulate_rng)
    logging.info(tabulate_fn(x))

    # train
    while state.step < config.train_steps:
        batch = next(data_iter)
        state, (loss, _) = train_step(state, batch)

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | loss {loss.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            logging.info(f'{Fore.GREEN}EVAL:{Style.RESET_ALL} loss {loss.item():.4f}')

        if state.step % config.ckpt_interval == 0:
            checkpoints.save_checkpoint(
                ckpt_dir, state, int(state.step), keep=float('inf')
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    jax.config.config_with_absl()
    app.run(main)
