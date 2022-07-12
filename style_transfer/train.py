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
from flax.core.frozen_dict import freeze, unfreeze
from flax.training import train_state
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


def load_vgg16_weights(model, params):
    """Load torchvision VGG16 weights into JAX params."""
    import torch
    from torchvision.models import vgg16

    torch_model = vgg16(pretrained=True)
    torch_params = torch_model.features.state_dict()

    mutable_params = unfreeze(params)
    for param_name in mutable_params['backbone'].keys():
        layer_id = param_name.split('_')[-1]

        torch_kernel = torch_params[f'{layer_id}.weight'].permute((2, 3, 1, 0))
        torch_bias = torch_params[f'{layer_id}.bias']

        mutable_params['backbone'][param_name]['kernel'] = jnp.array(torch_kernel)
        mutable_params['backbone'][param_name]['bias'] = jnp.array(torch_bias)
    new_params = freeze(mutable_params)

    # Test that the weights are loaded correctly
    jax_out = np.array(model.apply({'params': new_params}, jnp.ones((1, 32, 32, 3))))
    torch_out = (
        torch_model.features(torch.ones((1, 3, 32, 32)))
        .permute((0, 2, 3, 1))
        .detach()
        .numpy()
    )
    assert np.allclose(jax_out, torch_out, 1e-3)

    return new_params


class VGG16(nn.Module):
    """VGG16 (Simonyan et al., 2015) backbone.

    Max pooling operations replaced with average pooling operations as suggested in
    A Neural Algorithm of Artistic Style (Gatys et al., 2015).
    """

    def setup(self):
        self.backbone = nn.Sequential(
            [
                nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                functools.partial(
                    nn.max_pool,
                    window_shape=(2, 2),
                    strides=(2, 2),
                    padding=((0, 0), (0, 0)),
                ),
                nn.Conv(128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                functools.partial(
                    nn.max_pool,
                    window_shape=(2, 2),
                    strides=(2, 2),
                    padding=((0, 0), (0, 0)),
                ),
                nn.Conv(256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(256, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                functools.partial(
                    nn.max_pool,
                    window_shape=(2, 2),
                    strides=(2, 2),
                    padding=((0, 0), (0, 0)),
                ),
                nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                functools.partial(
                    nn.max_pool,
                    window_shape=(2, 2),
                    strides=(2, 2),
                    padding=((0, 0), (0, 0)),
                ),
                nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                nn.Conv(512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
                nn.relu,
                functools.partial(
                    nn.max_pool,
                    window_shape=(2, 2),
                    strides=(2, 2),
                    padding=((0, 0), (0, 0)),
                ),
            ]
        )

    def __call__(self, x):
        return self.backbone(x)


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


def create_state(rng, config):
    model = VGG16()

    # hold the model parameters fixed
    params = model.init(rng, jnp.ones((1, 32, 32, 3)))['params']
    params = load_vgg16_weights(model, params)

    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    # TODO: make the train state params the image
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


def main(argv):
    del argv  # Unused.

    rng = jax.random.PRNGKey(0)
    config = FLAGS.config
    np.random.seed(config.np_seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='neural_style-')

    examples_seen = 0

    # setup model and state
    rng, init_rng = jax.random.split(rng)
    state = create_state(init_rng, config)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    tabulate_fn = nn.tabulate(VGG16(), tabulate_rng)
    logging.info(tabulate_fn(jnp.ones((1, 32, 32, 3))))

    # train
    while state.step < config.train_steps:
        batch = next(data_iter)
        state, (loss, logits) = train_step(state, batch)

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
            recon = generate_image(state, 32, 32)
            img_dir = pathlib.Path(workdir) / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(img_dir / f'{state.step}.png')
            Image.fromarray(np.array(recon)).save(output_path)
            logging.info(
                f'{Fore.GREEN}EVAL:{Style.RESET_ALL} qualitative\n'
                f'{Fore.RED}ORIGINAL{Style.RESET_ALL}: {start_x}\n\n'
                f'{Fore.RED}PRED{Style.RESET_ALL}: {pred_x}\n\n'
            )


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
