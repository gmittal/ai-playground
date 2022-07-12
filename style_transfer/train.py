import functools
import os
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
from PIL import Image
from torch.utils.data import Dataset


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


def load_image(path, height, width):
    """Load and resize image."""
    img = Image.open(path)
    img = img.resize((height, width), Image.BILINEAR)
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    return img


def load_vgg16_weights(params):
    """Load torchvision VGG16 weights into JAX params."""
    from torchvision.models import vgg16

    torch_model = vgg16(pretrained=True)
    torch_params = torch_model.features.state_dict()

    mutable_params = unfreeze(params)
    for param_name in mutable_params.keys():
        layer_id = param_name.split('_')[-1]

        torch_kernel = torch_params[f'{layer_id}.weight'].permute((2, 3, 1, 0))
        torch_bias = torch_params[f'{layer_id}.bias']

        mutable_params[param_name]['kernel'] = jnp.array(torch_kernel)
        mutable_params[param_name]['bias'] = jnp.array(torch_bias)
    new_params = freeze(mutable_params)
    # TODO: add assert checking that max_pool jax implementation is correct

    return new_params


class VGG16(nn.Module):
    """VGG16 (Simonyan et al., 2015) backbone.

    Max pooling operations replaced with average pooling operations as suggested in
    A Neural Algorithm of Artistic Style (Gatys et al., 2015).
    """

    def setup(self):
        self.backbone = [
            nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            nn.relu,
            nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            nn.relu,
            functools.partial(
                nn.avg_pool,
                window_shape=(2, 2),
                strides=(2, 2),
                padding=((0, 0), (0, 0)),
            ),
            nn.Conv(128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            nn.relu,
            nn.Conv(128, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            nn.relu,
            functools.partial(
                nn.avg_pool,
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
                nn.avg_pool,
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
                nn.avg_pool,
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
                nn.avg_pool,
                window_shape=(2, 2),
                strides=(2, 2),
                padding=((0, 0), (0, 0)),
            ),
        ]

    def __call__(self, x, layer_idx=None):
        # normalization constants from torch
        mu = jnp.array([0.485, 0.456, 0.406])
        std = jnp.array([0.229, 0.224, 0.225])
        x = (x - mu) / std

        layer_activations = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if layer_idx is not None and i in layer_idx:
                layer_activations.append(x)
        return x if layer_idx is None else layer_activations


def create_state(init_image, config):
    model = VGG16()

    # hold the model parameters fixed
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)))['params']
    params = load_vgg16_weights(params)

    tx = optax.adam(config.learning_rate, config.beta1, config.beta2)
    image = freeze({'image': init_image})
    state = train_state.TrainState.create(apply_fn=model.apply, params=image, tx=tx)
    return state, params


def gram_matrix(act):
    b, h, w, c = act.shape
    act = act.transpose((0, 3, 1, 2))
    F = act.reshape(b * c, -1)
    return F @ F.T / (b * h * w * c)


@jax.jit
def transfer_step(
    *, state, vgg_params, content_image, style_image, content_weight, style_weight
):
    def loss_fn(params):
        input_image = jnp.clip(params['image'][None, :], 0, 1)
        input_act = state.apply_fn(
            {'params': vgg_params},
            input_image,
            layer_idx=[0 + 1, 5 + 1, 10 + 1, 19 + 1, 28 + 1],
        )
        content_act = state.apply_fn(
            {'params': vgg_params},
            content_image[None, :],
            layer_idx=[19 + 1],
        )
        style_act = state.apply_fn(
            {'params': vgg_params},
            style_image[None, :],
            layer_idx=[0 + 1, 5 + 1, 10 + 1, 19 + 1, 28 + 1],
        )

        content_loss = sum(
            [
                jnp.mean(optax.l2_loss(i_a, c_a))
                for i_a, c_a in zip([input_act[3]], content_act)
            ]
        )
        style_loss = sum(
            [
                jnp.mean(optax.l2_loss(gram_matrix(i_a), gram_matrix(s_a)))
                for i_a, s_a in zip(input_act, style_act)
            ]
        )
        content_loss = content_loss * content_weight
        style_loss = style_loss * style_weight

        total_loss = content_loss + style_loss
        loss_dict = {'style': style_loss, 'content': content_loss}
        return total_loss, loss_dict

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, loss_dict), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, (loss, loss_dict)


def main(argv):
    del argv  # Unused.

    config = FLAGS.config
    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)
    workdir = FLAGS.workdir
    if workdir is None:
        workdir = tempfile.mkdtemp(prefix='neural_style-')

    # load the images
    content_img = load_image(os.path.expanduser(config.content), 256, 256)
    style_img = load_image(os.path.expanduser(config.style), 256, 256)

    # setup model and state
    state, nn_params = create_state(content_img, config)

    # print model
    rng, tabulate_rng = jax.random.split(rng)
    tabulate_fn = nn.tabulate(VGG16(), tabulate_rng)
    logging.info(tabulate_fn(jnp.ones((1, 32, 32, 3))))

    while state.step < config.train_steps:
        state, (_, loss_dict) = transfer_step(
            state=state,
            vgg_params=nn_params,
            content_image=content_img,
            style_image=style_img,
            content_weight=1.0,
            style_weight=1e6,
        )

        if state.step % config.logging_interval == 0:
            closs = loss_dict['content']
            sloss = loss_dict['style']
            logging.info(
                f'step {state.step} | content_loss {closs.item():.4f} | '
                f'style_loss {sloss.item():.4f}'
            )

        if state.step % config.eval_interval == 0:
            img_dir = pathlib.Path(workdir) / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(img_dir / f'{state.step}.png')
            Image.fromarray(
                np.array(jnp.clip(state.params['image'], 0, 1) * 255).astype(np.uint8)
            ).save(output_path)
            logging.info(f'{Fore.GREEN}EVAL:{Style.RESET_ALL} {output_path}')


if __name__ == '__main__':
    flags.mark_flags_as_required(['config'])
    jax.config.config_with_absl()
    app.run(main)
