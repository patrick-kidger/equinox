import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
from torch.utils import data
from torchvision.datasets import MNIST

import equinox as eqx
from equinox import nn


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class Cast(object):
    def __call__(self, pic):
        out = np.array(pic, dtype=jnp.float32)
        out = out.reshape((1,) + out.shape)
        return out


class CNN(eqx.Module):
    layers: list
    conv_layer: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)

        self.conv_layer = [nn.Conv2d(1, 2, 3, key=key3), nn.Pool2d_Max((2, 2))]

        self.layers = [nn.Linear(338, 256, key=key1), nn.Linear(256, 10, key=key2)]

    def __call__(self, x):
        # x = x.resize((1,)+x.shape)
        for layer in self.conv_layer:
            x = layer(x)

        x = jnp.ravel(x)

        for layer in self.layers:
            x = layer(x)
            x = jnn.relu(x)

        return x


cnn = CNN(jrandom.PRNGKey(0))


@jax.vmap
@jax.value_and_grad
def loss_grad(x, y):
    return jax.numpy.mean((y - cnn(x)) ** 2)


# Define our dataset, using torch datasets
mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=Cast())
training_generator = NumpyLoader(
    mnist_dataset, shuffle=True, batch_size=64, num_workers=6
)


# Get the full train dataset (for checking accuracy while training)
# train_images = np.array(mnist_dataset.train_data)
# train_labels = one_hot(np.array(mnist_dataset.train_labels), 10)

# Get full test dataset
# mnist_dataset_test = MNIST('/tmp/mnist/', download=True, train=False)
# test_images = jnp.array(mnist_dataset_test.test_data.numpy(), dtype=jnp.float32)
# test_labels = one_hot(np.array(mnist_dataset_test.test_labels), 10)
# print(test_labels[0])


# train_img = jnp.array(train_images[0],dtype=jnp.float32)

# train_images = jnp.array(train_images,dtype=jnp.float32)
# train_images = train_images.reshape((train_images.shape[0],1)+train_images.shape[1:])
# print(train_images.shape)
# test_images = jnp.array(test_images,dtype=jnp.float32)
# test_images = test_images.reshape((test_images.shape[0],1)+test_images.shape[1:])
# print(test_images.shape)


lr = 0.001


def training_loop(model, dataloader, lr):
    optim = optax.adam(lr)
    opt_state = optim.init(model)
    for i, (X, Y) in enumerate(dataloader):
        print(X.shape)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        loss, grads = loss_grad(X, Y)
        print(loss.item())
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)


epochs = 50

for i in range(epochs):
    training_loop(cnn, training_generator, 0.001)

# out = cnn(test_img.reshape((1,)+test_img.shape))
# print(out)


# x_key, y_key, model_key = jax.random.split(jax.random.PRNGKey(0), 3)
# x = jrandom.normal(x_key, (100, 2))
# y = jrandom.normal(y_key, (100, 2))
# model = MyModule(model_key)
# grads = loss(model, x, y)
# learning_rate = 0.1
# model = jax.tree_map(lambda m, g: m - learning_rate * g, model, grads)
