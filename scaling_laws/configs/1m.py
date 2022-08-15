from configs import base


def get_config():
    config = base.get_config()

    # 1M params
    config.emb_dim = 128
    config.n_blocks = 5
    config.n_heads = 8
    config.learning_rate = 1e-3

    return config
