import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.np_seed = 42

    # optimizer
    config.learning_rate = 1e-3
    config.beta1 = 0.9
    config.beta2 = 0.98
    config.weight_decay = 1.0
    config.batch_size = 512
    config.train_steps = 100_000

    # model
    config.emb_dim = 128
    config.n_blocks = 2
    config.n_heads = 4
    config.block_size = 4

    config.emb_dropout_prob = 0.1
    config.attn_dropout_prob = 0.1
    config.block_dropout_prob = 0.1

    # dataset
    config.p = 97
    config.train_frac = 0.4

    # dataloader
    config.num_workers = 0

    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 100

    return config
