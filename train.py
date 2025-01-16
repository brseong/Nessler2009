from functools import partial
import pdb
import torch as th
from utils.Nessler2009 import Nessler2009
from utils.coding import encode_data
from jaxtyping import UInt8
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

num_steps = 50
population = 2
batch_size = 32
num_workers = 4
out_features = 10

SpikeLoader = DataLoader[Dataset[UInt8[th.Tensor, "Population 28 28"]]]

if __name__ == "__main__":
    encode_data = partial(encode_data, num_steps=num_steps, population=population)
    data_train = MNIST(".", download=True, train=True, transform=encode_data)
    data_test = MNIST(".", download=True, train=False, transform=encode_data)
    train_loader = SpikeLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = SpikeLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    net = Nessler2009(28 * 28 * population, out_features)
    try:
        loader = iter(train_loader)
        x = next(loader)[0].view(batch_size, num_steps, -1)
        # x = data_train[0][0].view(1, num_steps, -1)
        net(x)
    except Exception as e:
        print(e)
        pdb.set_trace()
