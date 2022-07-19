import functools
import itertools
import pathlib
import tempfile

import chex
import colorama
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision
import wandb

from absl import app
from absl import flags
from absl import logging
from flax.core import frozen_dict
from flax.training import train_state
from flax.training import checkpoints
from ml_collections import config_flags
from torch.utils.data import DataLoader


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


class Encoder(nn.Module):
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(784, name='fc2')(z)
        return z


class VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))


class G(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return x


class D(nn.Module):
    def setup(self):
        pass

    def __call__(self, x):
        return x


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


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, config, rng):
    z_rng = jax.random.fold_in(rng, state.step)
    images = batch['x']

    def loss_fn(params):
        logits, mu, logvar = VAE(**config).apply(
            {'params': params},
            images,
            z_rng,
        )
        chex.assert_equal_shape([logits, images])
        bce_loss = binary_cross_entropy_with_logits(logits, images).mean()
        kld_loss = kl_divergence(mu, logvar).mean()
        losses = {'bce': bce_loss, 'kld': kld_loss}
        return bce_loss + kld_loss, losses

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, losses), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, losses)


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, config, z_rng):
    images = batch['x']

    logits, mu, logvar = VAE(**config).apply({'params': state.params}, images, z_rng)
    sum_bce_loss = binary_cross_entropy_with_logits(logits, images).sum()
    sum_kld_loss = kl_divergence(mu, logvar).sum()
    return sum_bce_loss, sum_kld_loss


def compute_metrics(state, test_batches, config, rng):
    bce_loss = 0
    kld_loss = 0
    examples_seen = 0
    for (images, labels) in test_batches:
        rng, z_rng = jax.random.split(rng)
        images = [np.array(im) / 255.0 for im in images]
        batch = {
            'x': jnp.array(images).reshape(len(images), -1),
            'y': jnp.array(labels),
        }
        partial_bce, partial_kld = eval_step(state, batch, config, z_rng)
        bce_loss += partial_bce.item()
        kld_loss += partial_kld.item()
        examples_seen += len(batch[list(batch.keys())[0]])
    return {'bce': bce_loss / examples_seen, 'kld': kld_loss / examples_seen}


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config
    np.random.seed(config.np_seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='just_ask-')
    if config.wandb:
        wandb.init(project='just_ask_for_generalization', config=config)

    # setup data
    train_dataset = torchvision.datasets.MNIST('/tmp/mnist', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST('/tmp/mnist', train=False, download=True)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=numpy_collate,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=numpy_collate,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    train_iter = itertools.cycle(train_dataloader)
    examples_seen = 0

    # create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0, end_value=config.learning_rate, transition_steps=100
    )
    constant_fn = optax.constant_schedule(config.learning_rate)
    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, constant_fn], boundaries=[100]
    )

    # setup model, optimizer, and state
    ckpt_dir = pathlib.Path(workdir) / 'checkpoints'
    rng, init_rng, fwd_rng = jax.random.split(rng, num=3)
    model_config = frozen_dict.FrozenDict(latents=20)
    model = VAE(**model_config)
    params = model.init(init_rng, jnp.ones([1, 28 * 28]), fwd_rng)['params']
    tx = optax.adamw(
        learning_rate_fn,
        weight_decay=0,
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng, fwd_rng = jax.random.split(rng, num=3)
    x = jnp.ones([1, 28 * 28])
    tabulate_fn = nn.tabulate(model, tabulate_rng)
    logging.info(tabulate_fn(x, fwd_rng))

    # train
    while state.step < config.train_steps:
        images, labels = next(train_iter)
        images = [np.array(im)[:, :] / 255.0 for im in images]
        batch = {
            'x': jnp.array(images).reshape(len(images), -1),
            'y': jnp.array(labels),
        }

        rng, fwd_rng = jax.random.split(rng)
        state, (loss, losses) = train_step(state, batch, model_config, fwd_rng)

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            lr = learning_rate_fn(state.step)
            bce_loss = losses['bce']
            kld_loss = losses['kld']
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | lr {lr:.4f} '
                f'loss {loss.item():.4f} | bce {bce_loss.item():.4f} | '
                f'kld {kld_loss.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            rng, eval_rng = jax.random.split(rng)
            losses = compute_metrics(state, test_dataloader, model_config, eval_rng)
            bce_loss = losses['bce']
            kld_loss = losses['kld']
            logging.info(
                f'{Fore.GREEN}EVAL:{Style.RESET_ALL} step {state.step} | '
                f'epoch {epoch_frac:.2f} | lr {lr:.4f} '
                f'loss {bce_loss+kld_loss:.4f} | bce {bce_loss:.4f} | '
                f'kld {kld_loss:.4f}'
            )

        if state.step % config.ckpt_interval == 0:
            checkpoints.save_checkpoint(
                ckpt_dir, state, int(state.step), keep=float('inf')
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
