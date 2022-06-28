import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import numpy as np
from torch.utils.data import Dataset, DataLoader
import optax
from PIL import Image


b_kernel = jnp.sqrt(20) * jnp.array(np.random.randn(256, 2))


def report_model(model):
    """Log number of trainable parameters and their memory footprint."""
    trainable_params = np.sum([param.size for param in jax.tree_leaves(model.params)])
    footprint_bytes = np.sum(
        [param.size * param.dtype.itemsize for param in jax.tree_leaves(model.params)]
    )

    print("Number of trainable paramters: {:,}".format(trainable_params))
    print("Memory footprint: {}MB".format(footprint_bytes / 2**20))


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class ImageDataset(Dataset):
    def __init__(self, img):
        # img.shape = (H, W, C)
        self.img = img
        self.H, self.W, _ = img.shape
        self.size = self.H * self.W

    def __len__(self):
        return self.size

    def _index_to_coord(self, idx):
        return idx // self.W, idx % self.W

    def get_features(self, idx):
        y, x = self._index_to_coord(idx)
        y = 2.0 * (y / self.H) - 1.0
        x = 2.0 * (x / self.W) - 1.0
        return np.array([y, x])

    def get_label(self, idx):
        return self.img[self._index_to_coord(idx)] / 255.0

    def __getitem__(self, idx):
        features = self.get_features(idx)
        label = self.get_label(idx)
        return features, label


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x_proj = 2.0 * jnp.pi * x @ b_kernel.T
        x = jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)
        x = nn.sigmoid(x)
        return x


@jax.jit
def train_step(state, inputs, outputs):
    def loss_fn(params):
        outputs_hat = MLP().apply({"params": params}, inputs)
        loss = jnp.mean(jnp.sum(jnp.square(outputs_hat - outputs), axis=-1))
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss


def generate_image(state, height, width):
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y, sparse=False)

    yy = 2.0 * (yy / height) - 1.0
    xx = 2.0 * (xx / width) - 1.0
    coords = np.stack((yy, xx), axis=2).reshape(-1, 2)

    z_flat = MLP().apply({"params": state.params}, coords)
    recon = z_flat.reshape(height, width, 3)
    recon = (recon * 255).astype(np.uint8)
    return recon


x = jnp.ones((128, 2))
tabulate_fn = nn.tabulate(MLP(), jax.random.PRNGKey(0))
print(tabulate_fn(x))


rng = jax.random.PRNGKey(1)
mlp = MLP()
params = mlp.init(rng, jnp.ones([1, 2]))["params"]
tx = optax.adam(1e-3)
state = train_state.TrainState.create(apply_fn=mlp.apply, params=params, tx=tx)
report_model(state)

img = np.array(Image.open("abbey_road.jpg"))
train_ds = ImageDataset(img)

# Fourier feature mapping
B_kernel = np.random.randn(256, 2)

# TODO: check droplast, figure how to make this infinite/iteration based instead of epochs
dataloader = DataLoader(
    train_ds, collate_fn=numpy_collate, shuffle=True, batch_size=len(train_ds)
)

# TODO: print iter number, loss, throughput (examples per second), fractional epoch number, grad norm if grad clipping is on
# TODO: print if there are dataloading errors the error rate
# make dataloader infinite and replace erroring examples with replacement
# If there is a model error, then just let it die

# add config support
# add grad clipping, weight decay, sync batchnorm, etc.
# add high loss mining script for data engine
#  - can also use embedding/slice identification technique from stanford
# add rho-loss auto data engine
# add inline verification data engine
# label cache
# add muTransfer and use it to test scaling laws
# add SAM
# add mosaicml stuff and ffcv dataloader for fast dataloading
# learning rate schedules (cosine annealing, constant step LR, etc.)
# hindsight logging for model training https://bobbyy.org/papers/2020_Flor_VLDB.pdf
# - is this actually helpful with normal checkpointing?


loss_vals = []
grad_norms = []
for epoch in range(100):
    print(epoch)
    for i, batch in enumerate(dataloader):
        x, y = batch
        x = jnp.array(x)
        y = jnp.array(y)

        grads, loss = train_step(state, x, y)
        # import pdb; pdb.set_trace()
        loss_vals.append(loss)
        state = state.apply_gradients(grads=grads)

import matplotlib.pyplot as plt

recon = generate_image(state, train_ds.H, train_ds.W)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(recon)
plt.show()

plt.plot(loss_vals)
plt.show()
