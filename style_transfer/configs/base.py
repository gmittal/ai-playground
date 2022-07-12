import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # random seeds
    config.seed = 42

    # optimizer
    config.learning_rate = 1e-2
    config.beta1 = 0.9
    config.beta2 = 0.999
    config.train_steps = 100

    # images
    config.content = 'abbey_road.jpg'
    config.style = 'starry.jpeg'

    config.logging_interval = 1
    config.eval_interval = 10

    return config
