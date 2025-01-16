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
batch_size = 1
num_workers = 4
out_features = 2  # 10 classes default
feature_map = {0: 0, 1: 3}
learning_rate = 1e-3
device = th.device("cuda:2" if th.cuda.is_available() else "cpu")

wandb.config.update(  # type: ignore
    {
        "num_steps": num_steps,
        "population": population,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "out_features": out_features,
        "learning_rate": learning_rate,
    }
)

SpikeLoader = DataLoader[Dataset[UInt8[th.Tensor, "Timesteps Population 28 28"]]]

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
    for epoch in range(num_epochs):
        data: UInt8[th.Tensor, "Batch Timesteps Population 28 28"]
        for i, (data, target) in tqdm(enumerate(iter(train_loader))):
            mask = th.zeros_like(target, dtype=th.bool)
            for k, v in feature_map.items():
                mask |= target == v
            data = data[mask, :]
            if data.shape[0] == 0:
                continue
            data = data.view(batch_size, num_steps, -1).to(device)
            target = target.to(device)
            net(data)
            if i % 10 == 0:
                for k in range(out_features):
                    evidence = net.prob_z2k[:, k]
                    # img = th.zeros((28, 28)).to(device)
                    # for j in range(population):
                    #     img += j / (population - 1) * evidence[j]
                    wandb.log(
                        {
                            f"img": wandb.Image(
                                data[0].sum(dim=0).view(2 * 28, 28).float()
                            ),
                            # f"prob_z2k_{k}": wandb.Image(img),
                            f"prob_z2k_{k}": wandb.Image(evidence.view(2 * 28, 28)),
                            f"prob_z_{k}": net.prob_z[k],
                        }
                    )
