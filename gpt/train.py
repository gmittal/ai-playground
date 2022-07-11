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

from absl import app
from absl import flags
from absl import logging
from flax.training import train_state
from flax.training import checkpoints
from ml_collections import config_flags
from tensorflow_probability.substrates import jax as tfp
from torch.utils.data import Dataset, DataLoader


tfd = tfp.distributions
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


class Attention(nn.Module):
    out_dim: int
    deterministic: bool

    def setup(self):
        self.q_net = nn.Dense(self.out_dim)
        self.k_net = nn.Dense(self.out_dim)
        self.v_net = nn.Dense(self.out_dim)

        self.proj = nn.Dense(self.out_dim)

        self.attn_drop = nn.Dropout(0.1, deterministic=self.deterministic)
        self.proj_drop = nn.Dropout(0.1, deterministic=self.deterministic)

    def __call__(self, x):
        B, T, C = x.shape
        N, D = (self.n_heads,)

        q = self.q_net(x).r
        k = self.k_net(x)
        v = self.v_net(x)

        w = q @ k.tranpose(1, 2) / jnp.sqrt(D)
        w = nn.softmax(w, axis=-1)
        attention = self.attn_drop(w @ v)
        # TODO: complete this
        return attention


class Block(nn.Module):
    emb_dim: int
    block_size: int
    n_heads: int
    dropout_prob: float
    decoder_mask: jnp.ndarray

    def setup(self):
        self.attention = nn.SelfAttention(num_heads=self.n_heads)
        self.mlp = nn.Sequential(
            [
                nn.Dense(4 * self.emb_dim),
                nn.gelu,
                nn.Dense(self.emb_dim),
                # nn.Dropout(self.dropout_prob, deterministic=self.deterministic),
            ]
        )
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x):
        B, T, _ = x.shape
        causal_mask = nn.make_causal_mask(jnp.ones((B, T)))
        x = x + self.attention(x, causal_mask)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class GPT(nn.Module):
    n_blocks: int = 4
    n_heads: int = 4

    block_size: int = 128  # context length
    deterministic: bool = True

    def setup(self):
        vocab_size = 65
        emb_dim = 32

        self.token_emb = nn.Embed(
            num_embeddings=vocab_size,
            features=emb_dim,
            embedding_init=nn.initializers.normal(stddev=1.0),
        )

        # TODO: currently learnable, experiment with sinusoidal
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=1 / jnp.sqrt(emb_dim)),
            (1, self.block_size, emb_dim),
        )
        self.dropout = nn.Dropout(0.1)

        decoder_mask = nn.make_causal_mask(jnp.ones((1, self.block_size)))
        blocks = [
            Block(emb_dim, self.block_size, self.n_heads, 0.1, decoder_mask)
            for _ in range(self.n_blocks)
        ]
        self.transformer = nn.Sequential(blocks)

        self.ln = nn.LayerNorm()
        self.head = nn.Dense(vocab_size)

    def __call__(self, x):
        _, t = x.shape

        emb_tokens = self.token_emb(x)
        emb_pos = self.pos_embedding[:, :t, :]
        x = emb_tokens + emb_pos

        x = self.transformer(x)
        x = self.ln(x)
        x = self.head(x)
        return x


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]

        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = np.array(dix[:-1], dtype=np.int64)
        y = np.array(dix[1:], dtype=np.int64)
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


def create_train_state(rng, config):
    model = GPT()
    params = model.init(rng, jnp.ones([1, config.block_size], dtype=np.int64))['params']

    # TODO: make optimizer configurable
    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    tokens = batch['x']
    next_tokens = batch['y']

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, tokens)
        chex.assert_equal_shape([logits[:, :, 0], next_tokens])
        loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=next_tokens
            )
        )
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits)


def top_k(logits, k):
    return jnp.argpartition(logits, -k, axis=-1)[:, -k:]


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5))
def sample(state, x, steps, temperature=1.0, sample=False, top_k=None):
    # TODO: implement nucleus sampling
    for _ in range(steps):
        logits = state.apply_fn({'params': state.params}, x)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k(logits, top_k)
        probs = nn.softmax(axis=-1)
        if sample:
            x = jnp.random.choice(logits.shape[-1], p=probs)


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config
    np.random.seed(config.np_seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='gpt-jax_')

    # setup data
    text_data = open(config.input_file).read()
    train_dataset = CharDataset(text_data, config.block_size)
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
    ckpt_dir = pathlib.Path(workdir) / 'checkpoints'
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    x = train_dataset[0]['x'][None, :]
    tabulate_fn = nn.tabulate(GPT(), tabulate_rng)
    logging.info(tabulate_fn(x))

    # train
    while state.step < config.train_steps:
        batch = next(data_iter)
        state, (loss, logits) = train_step(state, batch)

        # additional metrics
        pred_tokens = logits.argmax(-1)
        acc = (pred_tokens == batch['y']).mean()

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | loss {loss.item():.4f} '
                f'| accuarcy {acc.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            import random

            idx = int(random.random() * len(batch['x']))
            i2s = lambda x: ''.join([train_dataset.itos[i] for i in x])
            start_x = i2s(batch['x'][idx])
            pred_x = i2s(np.array(logits.argmax(axis=-1)[idx]))
            # recon = generate_image(state, 32, 32)
            # img_dir = pathlib.Path(workdir) / 'images'
            # img_dir.mkdir(parents=True, exist_ok=True)
            # output_path = str(img_dir / f'{state.step}.png')
            # Image.fromarray(np.array(recon)).save(output_path)
            logging.info(
                f'{Fore.GREEN}EVAL:{Style.RESET_ALL} qualitative\n'
                f'{Fore.RED}ORIGINAL{Style.RESET_ALL}: {start_x}\n\n'
                f'{Fore.RED}PRED{Style.RESET_ALL}: {pred_x}\n\n'
            )

        if state.step % config.ckpt_interval == 0:
            checkpoints.save_checkpoint(
                ckpt_dir, state, int(state.step), keep=float('inf')
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
