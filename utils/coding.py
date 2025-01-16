import pdb
import torch as th
from torchvision.transforms import ToTensor  # type: ignore
from PIL.Image import Image
from jaxtyping import Float, UInt8, Int

to_tensor = ToTensor()


def poisson_spike_per_cls(
    y_classes: UInt8[th.Tensor, "1 28 28"],
    t_steps: int = 50,
    prob: float = 4e-2,
    population: int = 2,
) -> UInt8[th.Tensor, "50 Pop 28 28"]:
    """Generate Poisson spikes for each class in the input tensor. Only one class is active for the series of spikes.

    Returns:
        _type_: _description_
    """
    y_indices = y_classes.unsqueeze_(0).repeat(t_steps, 1, 1, 1)  # (t_steps, 1, 28, 28)
    poisson = th.distributions.poisson.Poisson(rate=prob)
    spikes = th.zeros(
        (t_steps, population, *y_indices.shape[2:]), dtype=th.uint8
    )  # (t_steps, Population, 28, 28)
    _spikes = poisson.sample((t_steps, 1, *y_indices.shape[2:])).to(
        th.uint8
    )  # (t_steps, 1, 28, 28)
    return spikes.scatter_(dim=1, index=y_indices, src=_spikes)


def encode_data(
    x: Image | th.Tensor, population: int = 2
) -> UInt8[th.Tensor, "Population 28 28"]:
    if not isinstance(x, th.Tensor):
        x = to_tensor(x)
    x *= population  # (1, 28, 28)
    x = (
        x.floor_().clamp_(0, population - 1).to(dtype=th.int64)
    )  # Split the image into population parts: 0,...,population-1

    return poisson_spike_per_cls(x)


if __name__ == "__main__":
    from torchvision.datasets import MNIST  # type: ignore

    mnist = MNIST("../", download=True, transform=encode_data)
    y = mnist[0][0]
    pdb.set_trace()
