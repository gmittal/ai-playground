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
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast

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

# use FPT initializations
Dense = functools.partial(
    nn.Dense,
    kernel_init=nn.initializers.orthogonal(scale=1.41),
    bias_init=nn.initializers.zeros,
)


class CausalSelfAttention(nn.Module):
    """Causal self-attention."""

    num_heads: int
    dropout_rate: float
    deterministic: bool

    @nn.compact
    def __call__(self, x):
        B, T, emb_dim = x.shape

        # compute query, key, value
        qkv = Dense(3 * emb_dim)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        head_dim = emb_dim // self.num_heads
        k = k.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q = q.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, head_dim).transpose(0, 2, 1, 3)

        # compute self-similarity + causal masking
        causal_mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]
        attn = q @ k.transpose(0, 1, 3, 2) * (1.0 / jnp.sqrt(head_dim))
        attn = jnp.where(causal_mask == 0, float('-inf'), attn)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(attn)

        y = attn @ v
        y = y.transpose(0, 2, 1, 3).reshape(B, T, emb_dim)
        y = Dense(emb_dim)(y)
        y = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(y)
        return y


class Block(nn.Module):
    emb_dim: int
    block_size: int
    n_heads: int

    residual_dropout_prob: float
    attn_dropout_prob: float
    deterministic: bool

    n_blocks: int = 1  # for residual projection initialization

    def setup(self):
        self.attention = CausalSelfAttention(
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

    def __call__(self, x):
        B, T, _ = x.shape
        x = x + self.attention(x)
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
        # TODO: try sinusoidal embedding initializer
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
                n_blocks=self.n_blocks,  # for residual projection initialization
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


# class CharDataset(Dataset):
#     def __init__(self, data, block_size):
#         self.bpe_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
#         self.block_size = block_size
#         self.vocab_size = self.bpe_tokenizer.vocab_size
#         self.data = np.array(self.bpe_tokenizer(data)['input_ids'], dtype=np.int32)

#     def __len__(self):
#         return len(self.data) - self.block_size

#     def encode(self, chunk):
#         return np.array(self.bpe_tokenizer(chunk)['input_ids'], dtype=np.int32)

#     def decode(self, x):
#         return self.bpe_tokenizer.decode(x)

#     def __getitem__(self, idx):
#         # grab a chunk of (block_size + 1) characters from the data
#         chunk = self.data[idx : idx + self.block_size + 1]
#         return {'x': chunk[:-1], 'y': chunk[1:]}


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
        return 'attention' in param_name or 'ln' in param_name or 'head' in param_name or 'token_emb' in param_name

    def count_filtered_params(t, prefix=''):
        count = 0
        for pn in t.keys():
            if isinstance(t[pn], frozen_dict.FrozenDict):
                count += count_filtered_params(t[pn], prefix=f'{prefix + pn}/')
            elif filter_fn(prefix + pn):
                count += t[pn].size
        return count

    print(f'Trainable parameters: {count_filtered_params(p)}')

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


# def create_weight_decay_param_mask(p):
#     def filter_fn(param_name):
#         # avoid all biases, layer norms, and embeddings
#         if (
#             param_name.endswith('bias')
#             or 'ln' in param_name
#             or param_name.endswith('embedding')
#         ):
#             return False

#         # everything else should be fine
#         return True

#     p = flax.traverse_util.ModelParamTraversal(lambda x, _: filter_fn(x)).update(
#         lambda _: True, p
#     )
#     p = flax.traverse_util.ModelParamTraversal(lambda x, _: not filter_fn(x)).update(
#         lambda _: False, p
#     )
#     return p


@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, config, rng):
    tokens = batch['x']
    next_tokens = batch['y']
    rng = jax.random.fold_in(rng, state.step)

    def loss_fn(params):
        logits = Transformer(**config, deterministic=False).apply(
            {'params': params},
            tokens,
            rngs={'dropout': rng},
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
    grads = jax.lax.pmean(grads, axis_name='batch')  # synchronize device gradients
    loss = jax.lax.pmean(loss, axis_name='batch')  # for logging
    state = state.apply_gradients(grads=grads)
    return state, (loss, logits), rng


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


@functools.partial(jax.jit, static_argnums=(2, 3, 5, 6, 7))
def sample(state, prompt, steps, config, rng, temperature=1.0, top_k=None, top_p=0.9):
    """
    Autoregressive decoding from the model.

    Args:
        state: Optimized model parameters.
        prompt: Encoded sequences of indices to use as the prompt (B, T).
        steps: Number of tokens to generate.
        config: Model configuration.
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
        logits = Transformer(**config, deterministic=True).apply(
            {'params': state.params},
            jax.lax.dynamic_slice(tokens, (0, window_start), (B, block_size)),
        )

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
        drop_last=True,  # TODO: fix this with pad + shard + unpad
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
    # tx = optax.chain(
    #     optax.clip_by_global_norm(config.grad_norm_clip),
    #     optax.adamw(
    #         learning_rate_fn,
    #         b1=config.beta1,
    #         b2=config.beta2,
    #         weight_decay=config.weight_decay,
    #         mask=create_weight_decay_param_mask,
    #     ),
    # )
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

    # data-parallel setup
    num_devices = jax.device_count()
    p_train_step = jax.pmap(train_step, axis_name='batch', donate_argnums=(0,), static_broadcasted_argnums=(2,))
    repl_rng = jax.random.split(rng, num=num_devices)
    repl_state = flax.jax_utils.replicate(state)

    batch_size_per_device, ragged = divmod(config.batch_size, num_devices)
    if ragged:
        msg = "batch size must be divisible by device count, got {} and {}."
        raise ValueError(msg.format(config.batch_size, num_devices))

    while state.step < config.train_steps:
        batch = next(data_iter)

        # shard batch
        shape_prefix = (num_devices, batch_size_per_device)
        batch = {k: v.reshape(shape_prefix + v.shape[1:]) for k, v in batch.items()}

        repl_state, (loss, _), repl_rng = p_train_step(repl_state, batch, model_config, repl_rng)

        # unreplicate
        state = flax.jax_utils.unreplicate(repl_state)
        loss = jax.tree_map(lambda x: x[0], loss)
        loss = jax.device_get(loss)

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
                jnp.array(train_dataset.encode("What is the meaning of life?"))[None, :],
                200,
                model_config,
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
