import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 1e-3
    config.lr_warmup_steps = 0
    config.lr_cosine_decay = False

    config.beta1 = 0.9
    config.beta2 = 0.95
    config.grad_norm_clip = 1.0
    config.batch_size = 1024
    config.train_steps = 10_000

    # model
    config.emb_dim = 128
    config.n_blocks = 4
    config.n_heads = 4
    config.block_size = 4

    config.emb_dropout_prob = 0.1
    config.attn_dropout_prob = 0.1
    config.block_dropout_prob = 0.1

    # dataset
    config.p = 97
    config.train_frac = 0.8

    # dataloader
    config.num_workers = 0

    # logging
    config.wandb = False
    config.logging_interval = 100
    config.eval_interval = 500
    config.ckpt_interval = 1000

    return config
