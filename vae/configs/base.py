import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.np_seed = 42

    # optimizer
    config.learning_rate = 1e-3
    config.beta1 = 0.9
    config.beta2 = 0.98
    config.weight_decay = 0.0
    config.batch_size = 128
    config.train_steps = 100_000

    # model
    config.latents = 32

    # dataset
    config.p = 97
    config.train_frac = 0.4

    # dataloader
    config.num_workers = 0

    # logging
    config.wandb = False
    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 100

    return config
