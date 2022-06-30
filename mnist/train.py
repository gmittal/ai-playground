import itertools
import pathlib

import colorama
import chex
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import torchvision.datasets

from absl import app
from absl import flags
from absl import logging
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


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=256, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
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


def create_train_state(rng, config):
    model = MLP()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']

    # TODO: make optimizer configurable
    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    images = batch['x']
    labels = batch['y']

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images)
        one_hot = nn.one_hot(labels, num_classes=10)
        chex.assert_equal_shape([logits, one_hot])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def compute_metrics(state, dataset):
    im_labels = [(im, label) for im, label in dataset]
    images, labels = zip(*im_labels)
    images = jnp.array([np.array(im)[:, :, None] / 255.0 for im in images])
    labels = jnp.array(labels)

    one_hot = nn.one_hot(labels, num_classes=10)
    logits = state.apply_fn({'params': state.params}, images)

    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {
        'loss': loss,
        'accuracy': acc,
    }


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config
    np.random.seed(config.np_seed)

    # setup data
    train_dataset = torchvision.datasets.MNIST('/tmp/mnist', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST('/tmp/mnist', train=False, download=True)
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
    x = np.array(train_dataset[0][0])[:, :, None]
    tabulate_fn = nn.tabulate(MLP(), tabulate_rng)
    logging.info(tabulate_fn(x))

    # train
    while state.step < config.train_steps:
        images, labels = next(data_iter)
        images = [np.array(im)[:, :, None] / 255.0 for im in images]
        batch = {'x': jnp.array(images), 'y': jnp.array(labels)}
        state, loss = train_step(state, batch)

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | loss {loss.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            metrics = compute_metrics(state, test_dataset)
            test_loss = metrics['loss']
            test_acc = metrics['accuracy']
            logging.info(
                f'{Fore.GREEN}EVAL:{Style.RESET_ALL} loss {test_loss:.4f} '
                f'| accuracy {test_acc:.4f}'
            )

        if state.step % config.ckpt_interval == 0:
            checkpoints.save_checkpoint(
                ckpt_dir, state, int(state.step), keep=float('inf')
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config', 'workdir'])
    jax.config.config_with_absl()
    app.run(main)
