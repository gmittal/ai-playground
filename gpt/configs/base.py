import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 3e-4
    config.beta1 = 0.9
    config.beta2 = 0.98
    config.weight_decay = 0.0
    config.batch_size = 512
    config.train_steps = 10_000

    # model
    config.emb_dim = 128
    config.n_blocks = 2
    config.n_heads = 4
    config.block_size = 4

    config.emb_dropout_prob = 0.1
    config.attn_dropout_prob = 0.1
    config.block_dropout_prob = 0.1

    # dataset
    config.data_file = 'shakespeare.txt'

    # dataloader
    config.num_workers = 0

    # logging
    config.wandb = False
    config.logging_interval = 10
    config.eval_interval = 100
    config.ckpt_interval = 5000

    return config
