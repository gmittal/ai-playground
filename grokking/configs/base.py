import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.np_seed = 42

    # optimizer
    config.learning_rate = 3e-4 * 4
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.batch_size = 64
    config.train_steps = 10_000

    # model
    config.block_size = 128

    # dataset
    config.input_file = 'shakespeare.txt'

    # dataloader
    config.num_workers = 0

    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 100

    return config
