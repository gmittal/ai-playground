import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 6e-4
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.weight_decay = 0.0
    config.batch_size = 512
    config.train_steps = 100_000

    # model
    config.emb_dim = 256
    config.n_blocks = 8
    config.n_heads = 8
    config.block_size = 128

    config.emb_dropout_prob = 0.1
    config.attn_dropout_prob = 0.1
    config.block_dropout_prob = 0.1

    # dataset
    config.data_file = 'shakespeare.txt'

    # dataloader
    config.num_workers = 0

    # logging
    config.wandb = False
    config.logging_interval = 100
    config.eval_interval = 500
    config.ckpt_interval = 1000

    return config
