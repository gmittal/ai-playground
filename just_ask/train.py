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
import wandb

from absl import app
from absl import flags
from absl import logging
from flax.core import frozen_dict
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
def train_step(state, batch, config, dropout_rng):
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    tokens = batch['x']
    next_tokens = batch['y']

    def loss_fn(params):
        logits = Transformer(**config, deterministic=False).apply(
            {'params': params},
            tokens,
            rngs={'dropout': dropout_rng},
        )
        chex.assert_equal_shape([logits[:, :, 0], next_tokens])
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=logits[:, -1, :], labels=next_tokens[:, -1]
            )
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits)


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, config):
    tokens = batch['x']
    next_tokens = batch['y']

    logits = Transformer(**config, deterministic=True).apply(
        {'params': state.params}, tokens
    )
    sum_loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(
            logits=logits[:, -1, :], labels=next_tokens[:, -1]
        )
    )
    pred_tokens = logits[:, -1, :].argmax(-1)
    sum_accuracy = jnp.sum(pred_tokens == next_tokens[:, -1])
    return sum_loss, sum_accuracy


def compute_metrics(state, test_batches, config):
    loss = 0
    accuracy = 0
    examples_seen = 0
    for batch in test_batches:
        partial_sum, partial_acc = eval_step(state, batch, config)
        loss += partial_sum.item()
        accuracy += partial_acc.item()
        examples_seen += len(batch[list(batch.keys())[0]])
    return {'loss': loss / examples_seen, 'accuracy': accuracy / examples_seen}


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
    train_dataloader = DataLoader(
        train_dataset,
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
        init_value=0.0, end_value=config.learning_rate, transition_steps=10
    )
    constant_fn = optax.constant_schedule(config.learning_rate)
    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, constant_fn], boundaries=[10]
    )

    # setup model, optimizer, and state
    ckpt_dir = pathlib.Path(workdir) / 'checkpoints'
    rng, init_rng = jax.random.split(rng)
    model_config = frozen_dict.FrozenDict(
        token_dim=config.p + 2,
        emb_dim=config.emb_dim,
        n_blocks=config.n_blocks,
        n_heads=config.n_heads,
        block_size=config.block_size,
        emb_dropout_prob=config.emb_dropout_prob,
        block_dropout_prob=config.block_dropout_prob,
        attn_dropout_prob=config.attn_dropout_prob,
    )
    model = Transformer(**model_config, deterministic=True)
    params = model.init(init_rng, jnp.ones([1, config.block_size], dtype=np.int64))[
        'params'
    ]
    tx = optax.adamw(
        learning_rate_fn,
        b1=config.beta1,
        b2=config.beta2,
        weight_decay=config.weight_decay,
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    x = train_dataset[0]['x'][None, :]
    tabulate_fn = nn.tabulate(model, tabulate_rng)
    logging.info(tabulate_fn(x))

    # train
    while state.step < config.train_steps:
        batch = next(train_iter)
        rng, dropout_rng = jax.random.split(rng)
        state, (loss, logits) = train_step(state, batch, model_config, dropout_rng)

        # additional training metrics
        pred_tokens = logits[:, -1, :].argmax(-1)
        acc = (pred_tokens == batch['y'][:, -1]).mean()

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            lr = learning_rate_fn(state.step)
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | lr {lr:.4f} '
                f'loss {loss.item():.4f} | accuracy {acc.item():.4f}'
            )
            wandb.log(
                {
                    'train': {
                        'lr': lr,
                        'loss': loss.item(),
                        'accuracy': acc.item(),
                        'epoch': epoch_frac,
                        'examples': examples_seen,
                    }
                },
                step=int(state.step),
            )

        if state.step % config.eval_interval == 0:
            metrics = compute_metrics(state, test_dataloader, model_config)
            val_loss = metrics['loss']
            val_acc = metrics['accuracy']
            logging.info(
                f'{Fore.GREEN}EVAL:{Style.RESET_ALL} step {state.step} | epoch '
                f'{epoch_frac:.2f} | loss {val_loss:.4f} | '
                f'accuracy {val_acc:.4f}'
            )
            wandb.log(
                {
                    'val': {
                        'loss': val_loss,
                        'accuracy': val_acc,
                    }
                },
                step=int(state.step),
            )

        if state.step % config.ckpt_interval == 0:
            checkpoints.save_checkpoint(
                ckpt_dir, state, int(state.step), keep=float('inf')
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
