from dataclasses import dataclass
import itertools
import pathlib

import colorama
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
from torch.utils.data import Dataset, DataLoader

Fore = colorama.Fore
Style = colorama.Style

FLAGS = flags.FLAGS
flags.DEFINE_string('config', None, 'Task configuration')
flags.mark_flag_as_required('config')


@dataclass
class Config:
    # optimizer
    lr = 3e-4
    bs = 512
    beta1 = 0.9
    beta2 = 0.99
    train_steps = 5_000

    # dataloader
    num_workers = 0

    eval_interval = 100
    ckpt_interval = 500
    workdir = './output'


class MLP(nn.Module):
    def setup(self):
        self.layers = [
            nn.Dense(4096),
            nn.Dense(1),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ToyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        mask = np.random.binomial(1, 0.8, size=(size,))[:, None]
        g1 = np.random.randn(size, 2) + 5
        g2 = np.random.randn(size, 2) - 5
        self.ds = np.float32(g1 * mask + g2 * (1 - mask))
        self.ds_y = mask.reshape(size, -1)

        self.ds_min = self.ds.min()
        self.ds_max = self.ds.max()

    def __len__(self):
        return self.size

    def _normalize(self, pt):
        return (pt - self.ds_min) / (self.ds_max - self.ds_min)

    def __getitem__(self, idx):
        return {'x': self._normalize(self.ds[idx]), 'y': self.ds_y[idx]}


def create_train_state(rng, config):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 2]))['params']

    # TODO: make optimizer configurable
    tx = optax.adam(config.lr, config.beta1, config.beta2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    images = batch['x']
    labels = batch['y']

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
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

    # setup data
    train_dataset = ToyDataset(size=10000)
    dataloader = DataLoader(
        train_dataset,
        collate_fn=numpy_collate,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        batch_size=Config.bs,
        num_workers=Config.num_workers,
    )
    data_iter = itertools.cycle(dataloader)
    examples_seen = 0

    # setup model and state
    ckpt_dir = pathlib.Path(Config.workdir) / 'checkpoints'
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, Config)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    while state.step < Config.train_steps:
        batch = next(data_iter)
        state, (loss, _) = train_step(state, batch)

        examples_seen += len(batch['x'])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % 10 == 0:
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | '
                f'loss {loss.item():.4f}'
            )

        if state.step % Config.eval_interval == 0:
            logging.info(f'{Fore.GREEN}EVAL:{Style.RESET_ALL} loss {loss.item():.4f}')

        if state.step % Config.ckpt_interval == 0:
            checkpoints.save_checkpoint(ckpt_dir, state, int(state.step), keep=-1)


if __name__ == '__main__':
    jax.config.config_with_absl()
    app.run(main)
