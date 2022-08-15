from configs import base


def get_config():
    config = base.get_config()

    # ~100K params
    # config.emb_dim = 64
    # config.n_blocks = 2
    # config.n_heads = 8
    # config.learning_rate = 1.5e-3

    # 1M params
    # config.emb_dim = 128
    # config.n_blocks = 5
    # config.n_heads = 8
    # config.learning_rate = 1e-3

    # ~10M
    # config.emb_dim = 256
    # config.n_blocks = 13
    # config.n_heads = 8
    # config.learning_rate = 1e-3

    # ~100M
    # config.emb_dim = 768
    # config.n_blocks = 14
    # config.n_heads = 12
    # config.learning_rate = 6.7e-4

    # ~1.5B (GPT-2)
    # config.emb_dim = 1600
    # config.n_blocks = 48
    # config.n_heads = 25
    # config.learning_rate = 3e-4

    return config
