import torch as th
from torch.nn import Module
from jaxtyping import UInt8


class Nessler2009(Module):
    last_input: UInt8[th.Tensor, "Batch Timesteps Population*28*28"]
    # last_winners: UInt8[th.Tensor, "Batch Timesteps out_features"]
    last_winner: UInt8[th.Tensor, "Batch Timesteps out_features"]

    def __init__(self, in_features: int, out_featuers: int = 10) -> None:
        super(Nessler2009, self).__init__()  # type: ignore

        self.log_prob_z2k = -th.abs(th.rand((in_features, out_featuers)))
        self.log_prob_z = -th.abs(th.rand((out_featuers)))
        self.time_window = 10

    def forward(self, x: UInt8[th.Tensor, "Batch Timesteps Population*28*28"]) -> None:
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Timesteps, Population*28*28)"
        )
        # self.last_input = x
        x_windowed = x[...]
        for t in range(1, self.time_window):
            x_windowed[:, t:, :] = x_windowed[:, t:, :] + x[:, :-t, :]
        x = x_windowed.sign_()

        for t in range(x.shape[1]):
            spk_train = x[:, t, :]
            self.last_input = x[:, t, :]
            bayes_prob = spk_train.float() @ th.exp(self.log_prob_z2k) + th.exp(
                self.log_prob_z
            )  # (Batch, out_features)
            winners = bayes_prob.softmax(dim=1).multinomial(1)
            # print(winners)
            winners = th.nn.functional.one_hot(winners, 10)
            winners = th.sum(winners, dim=0)
            print(winners)
            # self.last_winners[:, t, :] = th.nn.functional.one_hot(winner, 10)
            self.STDP()

    def STDP(self):
        dw = (
            th.zeros_like(self.log_prob_z2k)
            .unsqueeze(0)
            .repeat(self.last_input.shape[0], 1, 1)
        )  # Calculate the weight change for each input of the batch
        dw[
            :,
            :,
        ]
