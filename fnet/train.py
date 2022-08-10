import functools
import itertools
import pathlib
import tempfile

import chex
import colorama
import distrax
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
from scipy import linalg
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

# use GPT initializations
Dense = functools.partial(
    nn.Dense,
    kernel_init=nn.initializers.normal(stddev=0.02),
    bias_init=nn.initializers.zeros,
)


@jax.jit
def two_dim_matmul(x, matrix_dim_one, matrix_dim_two):
    """Applies 2D matrix multiplication to 2D input arrays."""
    return jnp.einsum(
        "ij,jk,ni->nk",
        x,
        matrix_dim_two,
        matrix_dim_one,
        optimize=True,
        precision=jax.lax.Precision.DEFAULT,
    )


class CausalFourierMixer(nn.Module):
    @nn.compact
    def __call__(self, x, dft_mat_seq=None, dft_mat_hidden=None):
        assert dft_mat_seq is not None and dft_mat_hidden is not None
        matmul = jax.vmap(
            functools.partial(
                two_dim_matmul,
                matrix_dim_one=dft_mat_seq,
                matrix_dim_two=dft_mat_hidden,
            )
        )
        return matmul(x).real


class Block(nn.Module):
    emb_dim: int
    block_size: int

    residual_dropout_prob: float
    deterministic: bool

    n_blocks: int = 1  # for residual projection initialization

    def setup(self):
        self.mixer = CausalFourierMixer()
        self.mlp = nn.Sequential(
            [
                Dense(4 * self.emb_dim),
                nn.gelu,
                nn.Dense(
                    self.emb_dim,
                    kernel_init=nn.initializers.normal(
                        stddev=0.02 / jnp.sqrt(2 * self.n_blocks)
                    ),
                    bias_init=nn.initializers.zeros,
                ),
                nn.Dropout(
                    self.residual_dropout_prob, deterministic=self.deterministic
                ),
            ]
        )
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()

    def __call__(self, x, dft_mat_seq=None, dft_mat_hidden=None):
        x = x + self.mixer(
            x, dft_mat_seq=dft_mat_seq, dft_mat_hidden=dft_mat_hidden
        )
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class FNet(nn.Module):
    token_dim: int
    emb_dim: int

    n_blocks: int
    block_size: int

    emb_dropout_prob: float
    block_dropout_prob: float
    deterministic: bool

    def setup(self):
        self.token_emb = nn.Embed(
            num_embeddings=self.token_dim,
            features=self.emb_dim,
            embedding_init=nn.initializers.normal(stddev=0.02),
        )
        # TODO: try sinusoidal embedding initializer
        self.pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=1 / jnp.sqrt(self.emb_dim)),
            (1, self.block_size, self.emb_dim),
        )
        self.dropout = nn.Dropout(
            self.emb_dropout_prob, deterministic=self.deterministic
        )

        self.blocks = [
            Block(
                emb_dim=self.emb_dim,
                block_size=self.block_size,
                n_blocks=self.n_blocks,  # for residual projection initialization
                residual_dropout_prob=self.block_dropout_prob,
                deterministic=self.deterministic,
            )
            for _ in range(self.n_blocks)
        ]

        self.ln = nn.LayerNorm()
        self.head = Dense(self.token_dim)

    def __call__(self, x, dft_mat_seq=None, dft_mat_hidden=None):
        _, t = x.shape

        emb_tokens = self.token_emb(x)
        emb_pos = self.pos_embedding[:, :t, :]
        x = emb_tokens + emb_pos
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, dft_mat_seq=dft_mat_seq, dft_mat_hidden=dft_mat_hidden)
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

    def encode(self, chunk):
        # Use int32 for now due to https://github.com/google/jax#current-gotchas
        return np.array([self.stoi[s] for s in chunk], dtype=np.int32)

    def decode(self, x):
        return ''.join([self.itos[i] for i in x])

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx : idx + self.block_size + 1]

        # encode every character to an integer
        dix = self.encode(chunk)
        return {'x': dix[:-1], 'y': dix[1:]}


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


def create_weight_decay_param_mask(p):
    def filter_fn(param_name):
        # avoid all biases, layer norms, and embeddings
        if (
            param_name.endswith('bias')
            or 'ln' in param_name
            or param_name.endswith('embedding')
        ):
            return False

        # everything else should be fine
        return True

    p = flax.traverse_util.ModelParamTraversal(lambda x, _: filter_fn(x)).update(
        lambda _: True, p
    )
    p = flax.traverse_util.ModelParamTraversal(lambda x, _: not filter_fn(x)).update(
        lambda _: False, p
    )
    return p


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, config, dft_mat_seq, dft_mat_hidden, dropout_rng):
    tokens = batch['x']
    next_tokens = batch['y']
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        logits = FNet(**config, deterministic=False).apply(
            {'params': params},
            tokens,
            dft_mat_seq,
            dft_mat_hidden,
            rngs={'dropout': dropout_rng},
        )
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


def top_k_logits(logits, k):
    B, _ = logits.shape
    topk_idx = jnp.argsort(-logits, axis=-1)[:, :k]
    rows, _ = jnp.indices((B, k))
    k_vals = jnp.min(logits[rows, topk_idx], axis=-1)
    return jnp.where(logits < k_vals[:, None], float('-inf'), logits)


def top_p_logits(logits, p):
    """Nucleus sampling"""
    B, C = logits.shape
    sorted_idx = jnp.argsort(-logits, axis=-1)
    rows, _ = jnp.indices((B, C))
    sorted_logits = logits[rows, sorted_idx]
    cdf = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
    cutoff_idx = jnp.sum(cdf <= p, axis=-1)
    cutoff_vals = jnp.min(sorted_logits[rows, cutoff_idx[:, None]], axis=-1)
    return jnp.where(logits < cutoff_vals[:, None], float('-inf'), logits)


@functools.partial(jax.jit, static_argnums=(2, 3, 7, 8, 9))
def sample(
    state,
    prompt,
    steps,
    config,
    dft_mat_seq,
    dft_mat_hidden,
    rng,
    temperature=1.0,
    top_k=None,
    top_p=0.9,
):
    """
    Autoregressive decoding from the model.

    Args:
        state: Optimized model parameters.
        prompt: Encoded sequences of indices to use as the prompt (B, T).
        steps: Number of tokens to generate.
        config: Model configuration.
        dft_mat_seq: DFT matrix for the sequence.
        dft_mat_hidden: DFT matrix for the hidden state.
        rng: random number generator.
        temperature: Temperature to use for sampling.
        top_k: Top k logits used for sampling.
        top_p: Logits masked based on CDF accumulation used for nucleus sampling.

    Returns:
        A generated sequence of indices of shape (B, T + steps)
    """
    assert steps >= 0, 'steps must be >= 0'

    B, prompt_len = prompt.shape
    prompt = jnp.pad(prompt, ((0, 0), (0, steps)))  # shape (B, prompt_len + steps)
    block_size = config['block_size']

    def sample_step(i, tokens):
        window_start = jnp.where(i < block_size, 0, i - block_size)
        logits = FNet(**config, deterministic=True).apply(
            {'params': state.params},
            jax.lax.dynamic_slice(tokens, (0, window_start), (B, block_size)),
            dft_mat_seq,
            dft_mat_hidden,
        )

        # TODO: add <sos> token so we can generate without prompt
        # to predict the i-th token we must use the logit from the prev position
        logits = logits[:, jnp.where(i < block_size, i - 1, -1), :] / temperature
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        if top_p is not None:
            logits = top_p_logits(logits, top_p)

        sample_rng = jax.random.fold_in(rng, i)
        next_token_dist = distrax.Categorical(logits=logits)
        next_token = next_token_dist.sample(seed=sample_rng)
        return tokens.at[:, i].set(next_token)

    seq = jax.lax.fori_loop(prompt_len, prompt_len + steps, sample_step, prompt)
    return seq


def train(config):
    rng = jax.random.PRNGKey(config.seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='gpt-')
    logging.info(f'workdir: {workdir}')
    if config.wandb:
        wandb.init(project='flax-gpt', config=config)

    # setup data pipeline
    text_data = open(config.data_file).read()
    train_dataset = CharDataset(text_data, config.block_size)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=numpy_collate,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    data_iter = itertools.cycle(train_dataloader)
    examples_seen = 0

    # setup model and optimizer
    rng, init_rng = jax.random.split(rng)
    model_config = frozen_dict.FrozenDict(
        token_dim=train_dataset.vocab_size,
        emb_dim=config.emb_dim,
        n_blocks=config.n_blocks,
        block_size=config.block_size,
        emb_dropout_prob=config.emb_dropout_prob,
        block_dropout_prob=config.block_dropout_prob,
    )

    # create causal-masked DFT
    dft_mat_seq = linalg.dft(config.block_size)
    for i in range(config.block_size):
        row = np.pad(linalg.dft(i + 1)[i, :], ((0), (config.block_size - (i + 1))))
        dft_mat_seq[i, :] = row
    dft_mat_seq = jnp.asarray(dft_mat_seq)
    dft_mat_hidden = jnp.asarray(linalg.dft(config.emb_dim))

    model = FNet(**model_config, deterministic=True)
    fake_sequence = jnp.ones([1, config.block_size], dtype=jnp.int32)
    params = model.init(init_rng, fake_sequence, dft_mat_seq, dft_mat_hidden)['params']
    learning_rate_fn = create_learning_rate_fn(config)
    tx = optax.chain(
        optax.clip_by_global_norm(config.grad_norm_clip),
        optax.adamw(
            learning_rate_fn,
            b1=config.beta1,
            b2=config.beta2,
            weight_decay=config.weight_decay,
            mask=create_weight_decay_param_mask,
        ),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    ckpt_dir = pathlib.Path(workdir) / 'checkpoints'
    state = checkpoints.restore_checkpoint(ckpt_dir, state)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    tabulate_fn = nn.tabulate(model, tabulate_rng)
    logging.info(tabulate_fn(fake_sequence, dft_mat_seq, dft_mat_hidden))

    while state.step < config.train_steps:
        batch = next(data_iter)
        rng, dropout_rng = jax.random.split(rng)
        state, (loss, _) = train_step(
            state, batch, model_config, dft_mat_seq, dft_mat_hidden, dropout_rng
        )

        examples_seen += len(batch[list(batch.keys())[0]])
        epoch_frac = examples_seen / len(train_dataset)

        if state.step % config.logging_interval == 0:
            lr = learning_rate_fn(state.step)
            logging.info(
                f'step {state.step} | epoch {epoch_frac:.2f} | lr {lr:.4f} | '
                f'loss {loss.item():.4f}'
            )
            if config.wandb:
                wandb.log(
                    {
                        'train': {
                            'lr': lr,
                            'loss': loss.item(),
                            'epoch': epoch_frac,
                            'examples': examples_seen,
                        }
                    },
                    step=int(state.step),
                )

        if state.step % config.eval_interval == 0:
            rng, sample_rng = jax.random.split(rng)
            seq = sample(
                state,
                jnp.array(train_dataset.encode("O God! O God!"))[None, :],
                2000,
                model_config,
                dft_mat_seq,
                dft_mat_hidden,
                sample_rng,
            )
            print(train_dataset.decode(np.array(seq[0])))

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
