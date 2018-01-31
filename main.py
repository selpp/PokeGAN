from train import train

train(
    dir = 'data/resized_images',
    random_dim = 500,
    width = 64,
    height = 64,
    channels = 3,
    batch_size = 64,
    epoch = 5000
)
