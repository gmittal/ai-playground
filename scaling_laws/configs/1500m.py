from configs import base


def get_config():
    config = base.get_config()

    # ~1.5B (GPT-2)
    config.emb_dim = 1600
    config.n_blocks = 48
    config.n_heads = 25
    config.learning_rate = 3e-4

    return config
