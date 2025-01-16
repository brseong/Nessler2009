from functools import partial
import pdb
import torch as th
from utils.Nessler2009 import Nessler2009
from utils.coding import encode_data
from jaxtyping import UInt8
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import wandb

wandb.init(project="nessler2009")

num_steps = 50
population = 2
num_epochs = 100
batch_size = 32
num_workers = 4
out_features = 10
learning_rate = 1e-1
device = th.device("cuda" if th.cuda.is_available() else "cpu")

SpikeLoader = DataLoader[Dataset[UInt8[th.Tensor, "Population 28 28"]]]

if __name__ == "__main__":
    th.no_grad()
    encode_data = partial(encode_data, num_steps=num_steps, population=population)
    data_train = MNIST(".", download=True, train=True, transform=encode_data)
    data_test = MNIST(".", download=True, train=False, transform=encode_data)
    train_loader = SpikeLoader(
        data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = SpikeLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    net = Nessler2009(28 * 28 * population, out_features, learning_rate).to(device)
    wandb.watch(net)  # type: ignore
    try:
        for epoch in range(num_epochs):
            for i, (data, target) in tqdm(enumerate(iter(train_loader))):
                data = data.view(batch_size, num_steps, -1).to(device)
                target = target.to(device)
                net(data)
                if i % 10 == 0:
                    for k in range(out_features):
                        evidence = net.prob_z2k.view(population, 28, 28, out_features)[
                            ..., k
                        ]
                        evidence = 0.5 + 0.5 * (evidence[1] - evidence[0])
                        wandb.log(
                            {
                                f"prob_z2k_{k}": wandb.Image(evidence),
                                f"prob_z_{k}": net.prob_z[k],
                            }
                        )
    except AssertionError as e:
        print(e)
        pdb.set_trace()
