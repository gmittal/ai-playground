"""
Pretrained Transformers As Universal Computation Engines
"""

import functools
import itertools
import pathlib
import tempfile

import chex
import colorama
import flax
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

Dense = functools.partial(
    nn.Dense,
    kernel_init=nn.initializers.orthogonal(scale=1.41),
    bias_init=nn.initializers.zeros,
)


class Block(nn.Module):
    emb_dim: int
    block_size: int
    n_heads: int

    residual_dropout_prob: float
    attn_dropout_prob: float
    deterministic: bool

    def setup(self):
        self.attention = nn.SelfAttention(
            num_heads=self.n_heads,
            dropout_rate=self.attn_dropout_prob,
            deterministic=self.deterministic,
        )
        self.mlp = nn.Sequential(
            [
                Dense(4 * self.emb_dim),
                nn.gelu,
                nn.Dense(
                    self.emb_dim,
                    kernel_init=nn.initializers.normal(stddev=0.02),
                    bias_init=nn.initializers.zeros,
                ),
                nn.Dropout(
                    self.residual_dropout_prob, deterministic=self.deterministic
                ),
            ]
        )
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x):
        B, T, _ = x.shape
        causal_mask = nn.make_causal_mask(jnp.ones((B, T)))
        x = x + self.attention(x)  # TODO: experiment with causal masking
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class Transformer(nn.Module):
    token_dim: int
    emb_dim: int

    n_blocks: int
    n_heads: int
    block_size: int

    emb_dropout_prob: float
    block_dropout_prob: float
    attn_dropout_prob: float
    deterministic: bool

    def setup(self):
        self.token_emb = nn.Embed(
            num_embeddings=self.token_dim,
            features=self.emb_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.emb_dim)),
            (1, self.block_size, self.emb_dim),
        )
        self.dropout = nn.Dropout(
            self.emb_dropout_prob, deterministic=self.deterministic
        )

        blocks = [
            Block(
                emb_dim=self.emb_dim,
                block_size=self.block_size,
                n_heads=self.n_heads,
                attn_dropout_prob=self.attn_dropout_prob,
                residual_dropout_prob=self.block_dropout_prob,
                deterministic=self.deterministic,
            )
            for _ in range(self.n_blocks)
        ]
        self.transformer = nn.Sequential(blocks)

        self.ln = nn.LayerNorm()
        self.head = Dense(self.token_dim)

    def __call__(self, x):
        _, t = x.shape

        emb_tokens = self.token_emb(x)
        emb_pos = self.pos_embedding[:, :t, :]
        x = emb_tokens + emb_pos
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.ln(x)
        x = self.head(x)
        return x


class BinaryOpDataset(Dataset):
    def __init__(self, *, p, is_train, train_frac, order):
        self.p = p
        self.binary_op = lambda a, b: (a * pow(b, p - 2, p))
        self.is_train = is_train
        self.train_frac = train_frac
        self.offset = 0 if is_train else int(train_frac * p * p)
        self.idx = order

        vocab = ['o', '='] + list(range(p))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

    def __len__(self):
        if self.is_train:
            return int(self.train_frac * self.p * self.p)
        return self.p * self.p - self.offset

    def __getitem__(self, idx):
        # a o b = binary_op(a, b) (mod p)
        idx += self.offset
        idx = self.idx[idx]
        a = idx % self.p
        b = idx // self.p
        assert 0 <= a < self.p and 0 <= b < self.p
        c = self.binary_op(a, b) % self.p
        eq = [a, 'o', b, '=', c]
        encoded_eq = [self.stoi[ch] for ch in eq]

        x = np.array(encoded_eq[:-1], dtype=np.int64)
        y = np.array(encoded_eq[1:], dtype=np.int64)
        return {'x': x, 'y': y}


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


def create_learning_rate_fn(config):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.lr_warmup_steps,
    )
    if config.lr_cosine_decay:
        decay_steps = config.train_steps - config.lr_warmup_steps
        opt_fn = optax.cosine_decay_schedule(
            init_value=config.learning_rate, decay_steps=decay_steps
        )
    else:
        opt_fn = optax.constant_schedule(config.learning_rate)

    learning_rate_fn = optax.join_schedules(
        schedules=[warmup_fn, opt_fn], boundaries=[config.lr_warmup_steps]
    )
    return learning_rate_fn


def create_frozen_param_mask(p):
    def filter_fn(param_name):
        # only fine-tune layernorms + output
        # TODO: maybe do pos embeddings too?
        return 'ln' in param_name or 'head' in param_name or 'token_emb' in param_name

    p = flax.traverse_util.ModelParamTraversal(lambda x, _: filter_fn(x)).update(
        lambda _: True, p
    )
    p = flax.traverse_util.ModelParamTraversal(lambda x, _: not filter_fn(x)).update(
        lambda _: False, p
    )
    return p


def zero_grads():
    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None):
        del state, params  # unused
        return jax.tree_map(jnp.zeros_like, updates), ()

    return optax.GradientTransformation(init_fn, update_fn)


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, config, dropout_rng):
    tokens = batch['x']
    next_tokens = batch['y']
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

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


def train(config):
    rng = jax.random.PRNGKey(config.seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='fpt-')
    logging.info(f'workdir: {workdir}')
    if config.wandb:
        wandb.init(project='flax-fpt', config=config)

    # setup data
    order = list(range(config.p * config.p))
    np.random.shuffle(order)
    train_dataset = BinaryOpDataset(
        p=config.p, is_train=True, train_frac=config.train_frac, order=order
    )
    test_dataset = BinaryOpDataset(
        p=config.p, is_train=False, train_frac=config.train_frac, order=order
    )
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

    # setup model and optimizer
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
    fake_sequence = jnp.ones([1, config.block_size], dtype=jnp.int32)
    params = model.init(init_rng, fake_sequence)['params']
    learning_rate_fn = create_learning_rate_fn(config)
    tx = optax.multi_transform(
        {
            True: optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adam(
                    learning_rate_fn,
                    b1=config.beta1,
                    b2=config.beta2,
                ),
            ),
            False: zero_grads(),
        },
        create_frozen_param_mask(params),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    ckpt_dir = pathlib.Path(workdir) / 'checkpoints'
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    tabulate_fn = nn.tabulate(model, tabulate_rng)
    logging.info(tabulate_fn(fake_sequence))

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
                f'step {state.step} | epoch {epoch_frac:.2f} | lr {lr:.4f} | '
                f'loss {loss.item():.4f} | accuracy {acc.item():.4f}'
            )
            if config.wandb:
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
            if config.wandb:
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

    return state


def main(argv):
    del argv  # Unused.

    config = FLAGS.config
    np.random.seed(config.seed)
    _ = train(config)


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
