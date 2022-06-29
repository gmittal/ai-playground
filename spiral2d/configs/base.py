import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # optimizer
    config.learning_rate = 1e-2
    config.beta1 = 0.9
    config.beta2 = 0.99
    config.batch_size = 128
    config.train_steps = 5_000

    # dataloader
    config.num_workers = 0

    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 500

    return config
