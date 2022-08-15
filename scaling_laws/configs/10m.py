from configs import base


def get_config():
    config = base.get_config()

    # ~10M
    config.emb_dim = 256
    config.n_blocks = 13
    config.n_heads = 8
    config.learning_rate = 1e-3

    return config
