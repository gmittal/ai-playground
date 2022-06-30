import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.np_seed = 42

    # optimizer
    config.learning_rate = 1e-2
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.batch_size = 128
    config.train_steps = 500

    # random kernel
    config.fourier_feat = True
    config.kernel_dim = 256
    config.initial_variance = 20.0

    # dataloader
    config.image_path = 'hendrix.jpg'
    config.num_workers = 0

    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 100

    return config
