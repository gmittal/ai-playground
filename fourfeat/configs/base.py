import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.np_seed = 42

    # optimizer
    # uses full batch gradient descent
    config.learning_rate = 3e-3
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.train_steps = 500

    # random kernel
    config.fourier_feat = True
    config.kernel_dim = 256
    config.initial_variance = 20.0

    # dataset/dataloader
    config.image_path = 'abbey_road.jpg'
    config.num_workers = 0

    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 100

    return config
