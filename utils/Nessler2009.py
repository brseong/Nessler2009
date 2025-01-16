import pdb
import torch as th
from torch.nn import Module
from jaxtyping import UInt8, Float
from matplotlib import pyplot as plt


class Nessler2009(Module):
    stdp_counter: int = 0
    prob_z2k: Float[th.Tensor, "in_features out_features"]  # noqa: F722
    prob_z: Float[th.Tensor, "out_features"]  # noqa: F821

    def __init__(
        self, in_features: int, out_features: int = 10, learning_rate: float = 1e-3
    ) -> None:
        super(Nessler2009, self).__init__()  # type: ignore

        self.in_features = in_features
        self.out_featuers = out_features
        self.learning_rate = learning_rate
        self.lr_decay_inverse = 1
        self.time_window = 10

        prob_z2k = th.rand((in_features, out_features))
        prob_z = th.rand((out_features))
        self.register_buffer("prob_z2k", prob_z2k)
        self.register_buffer("prob_z", prob_z)

    def forward(self, x: UInt8[th.Tensor, "Batch Timesteps Population*28*28"]) -> None:
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Timesteps, Population*28*28)"
        )
        # self.last_input = x
        x_windowed = x[...]
        for t in range(1, self.time_window):
            x_windowed[:, t:, :] = x_windowed[:, t:, :] + x[:, :-t, :]
        x = x_windowed.sign_()
        assert not x.isnan().any(), (
            "NaN in x_windowed: {}".format(x),
            pdb.set_trace(),
        )

        for t in range(x.shape[1]):
            step_spike = x[:, t, :]
            bayes_prob = (
                th.exp(step_spike.float()) @ self.prob_z2k * self.prob_z
            )  # (Batch, out_features)
            winners = bayes_prob.softmax(dim=1).multinomial(1).view(x.shape[0])
            winners = th.nn.functional.one_hot(winners, self.out_featuers).to(
                th.uint8
            )  # (Batch, out_features)
            self.STDP(step_spike, winners)

    def STDP(
        self,
        step_spike: UInt8[th.Tensor, "Batch in_features"],
        winners: UInt8[th.Tensor, "Batch out_features"],
    ) -> None:
        assert step_spike.dtype == winners.dtype, (
            f"{step_spike.dtype} != {winners.dtype}"
        )
        step_spike = step_spike.float()
        winners = winners.float()
        LTP_mask = step_spike.unsqueeze(-1) @ winners.unsqueeze(
            1
        )  # (Batch, in_features, out_features)
        LTD_mask = (1 - step_spike).unsqueeze(-1) @ winners.unsqueeze(1)

        dw = (self.prob_z2k - 1).unsqueeze_(0).repeat(
            step_spike.shape[0], 1, 1
        ) * LTP_mask - LTD_mask
        self.prob_z2k += (
            1
            / self.lr_decay_inverse
            * self.learning_rate
            * dw.mean(dim=0)
            * (self.prob_z2k)
            * (1 - self.prob_z2k)
        )  # mean over batch
        self.prob_z2k = th.nn.functional.normalize(
            self.prob_z2k, p=1, dim=0
        )  # Normalize over in_features, to satisfy the constraint that the sum of the weights to each output neuron is 1 (the weights is prob of the input spike given the latent variable)

        db = (self.prob_z - 1).unsqueeze_(0).repeat(
            step_spike.shape[0], 1
        ) * winners - (1 - winners)
        self.prob_z += (
            1
            / self.lr_decay_inverse
            * self.learning_rate
            * db.mean(dim=0)
            * (self.prob_z)
            * (1 - self.prob_z)
        )  # mean over batch
        self.prob_z = th.nn.functional.normalize(self.prob_z, p=1, dim=0)
        self.lr_decay_inverse += 1
