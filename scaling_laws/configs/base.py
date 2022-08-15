import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 5e-4
    config.lr_warmup_steps = 3_000
    config.lr_cosine_decay = True

    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.1
    config.grad_norm_clip = 1.0

    config.batch_size = 512
    config.train_steps = 250_000

    # model
    config.emb_dim = 192
    config.n_blocks = 6
    config.n_heads = 6
    config.block_size = 1024

    config.emb_dropout_prob = 0.1
    config.attn_dropout_prob = 0.1
    config.block_dropout_prob = 0.1

    # dataset
    config.data_file = 'data/shakespeare.txt'

    # dataloader
    config.num_workers = 0

    # logging
    config.wandb = False
    config.logging_interval = 1  # 50
    config.eval_interval = 500
    config.ckpt_interval = 1000

    return config
