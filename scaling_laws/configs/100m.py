from configs import base


def get_config():
    config = base.get_config()

    # ~100M
    config.emb_dim = 768
    config.n_blocks = 14
    config.n_heads = 12
    config.learning_rate = 6.7e-4

    return config
