from configs import base


def get_config():
    config = base.get_config()

    # ~100K params
    config.emb_dim = 64
    config.n_blocks = 2
    config.n_heads = 8
    config.learning_rate = 1.5e-3

    return config
