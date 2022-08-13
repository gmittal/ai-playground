from configs import base

def get_config():
    config = base.get_config()

    # optimizer
    config.learning_rate = 5e-4
    config.lr_warmup_steps = 3_000
    config.lr_cosine_decay = True

    config.batch_size = 64
    config.train_steps = 250_000

    # model
    config.emb_dim = 48
    config.n_blocks = 3
    config.n_heads = 3
    config.block_size = 128

    return config