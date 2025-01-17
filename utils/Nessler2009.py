import math
import pdb
from turtle import pd
import torch as th
from torch.nn import Module
from jaxtyping import UInt8, Float, Float64, Int
from matplotlib import pyplot as plt
import wandb


class Nessler2009(Module):
    log_likelihood: Float64[th.Tensor, "in_features out_features"]  # noqa: F722
    "Probability of the input spike given the latent variable, shape of: (populations * in_features, out_features)"
    log_prior: Float64[th.Tensor, "out_features"]  # noqa: F821
    "Marginal probability of the latent variable, shape of: (out_features, )"
    trace_pre_ltp: UInt8[th.Tensor, "Batch Timesteps in_features"]
    "Trace of LTP in presynaptic spike. type: UInt8[th.Tensor, 'Batch Timesteps in_features']"
    trace_pre_ltd: UInt8[th.Tensor, "Batch Timesteps in_features"]
    "Trace of LTD in presynaptic spike. type: UInt8[th.Tensor, 'Batch Timesteps in_features']"
    trace_post: UInt8[th.Tensor, "Batch Timesteps out_features"]
    "Trace of LTP in postsynaptic spike. type: UInt8[th.Tensor, 'Batch Timesteps out_features']"

    def __init__(
        self,
        in_features: int,
        out_features: int = 10,
        learning_rate: float = 1e-3,
        populations: int = 2,
        dtype: th.dtype = th.float64,
    ) -> None:
        super(Nessler2009, self).__init__()  # type: ignore

        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.inverse_lr_decay = 1
        "To garantee the convergence of the algorithm"
        self.populations = populations
        "Number of populations for population coding"
        self.sigma = 10
        "Time window for membrane potential and STDP"
        self.dtype = dtype

        log_likelihood = th.rand((in_features, out_features), dtype=dtype) * -1 - 1
        # likelihood = th.full((in_features, out_features), 0.5, dtype=th.float64)
        log_prior = th.rand((out_features,), dtype=dtype) * -1 - 1
        # prior = th.full((out_features,), 0.5, dtype=th.float64)
        self.register_buffer("log_likelihood", log_likelihood)
        self.register_buffer("log_prior", log_prior)
        self.normalize_probs()

    def forward(
        self, x: UInt8[th.Tensor, "Batch Num_steps Num_Population*28*28"]
    ) -> None:
        assert len(x.shape) == 3, (
            "Input shape must be (Batch, Num_steps, Num_Population*28*28)"
        )
        batch, num_steps, in_features = x.shape

        # Simulating the simplified membrane potential
        potential = x[...].bool()
        for t in range(1, self.sigma):
            potential[:, t:, :] |= x[:, :-t, :].bool()
        potential = potential.to(th.uint8)

        # To save the timings of the last spike for STDP
        self.trace_pre_ltp = th.zeros(
            (batch, num_steps, in_features), dtype=th.uint8, device=x.device
        )
        self.trace_pre_ltd = th.zeros(
            (batch, num_steps, in_features), dtype=th.uint8, device=x.device
        )
        self.trace_post = th.zeros(
            (batch, num_steps, self.out_features),
            dtype=th.uint8,
            device=x.device,
        )

        for t in range(potential.shape[1]):
            potential_t = potential[:, t, :]  # (Batch, in_features)
            posterior = th.exp(
                potential_t.double() @ self.log_likelihood + self.log_prior
            )  # (Batch, out_features)
            winners = (
                posterior.softmax(dim=1).multinomial(1).view(potential_t.shape[0])
            )  # (Batch, )
            winners = th.nn.functional.one_hot(winners, self.out_features).to(
                th.uint8
            )  # (Batch, out_features)
            # self.trace_pre -= 1
            # self.trace_pre[x[:, t, :].bool()] = 0
            # self.trace_post -= 1
            # self.trace_post[winners.bool()] = 0
            self.trace_pre_ltp[:, t : t + self.sigma, :] = (
                self.trace_pre_ltp[:, t : t + self.sigma, :] + x[:, t : t + 1, :]
            )  # (Batch, sigma, in_features). Broadcast the spike through the time dimension
            self.trace_pre_ltd[:, t + self.sigma :, :] = (
                self.trace_pre_ltd[:, t + self.sigma :, :] + x[:, t : t + 1, :]
            )
            self.trace_post[:, t:, :] = self.trace_post[:, t:, :] + winners.unsqueeze(1)
            self.STDP(x[:, t, :], winners, t)

    def STDP(
        self,
        pre_spikes: UInt8[th.Tensor, "Batch in_features"],
        post_spikes: UInt8[th.Tensor, "Batch out_features"],
        t: int,
    ) -> None:
        pre_spikes = pre_spikes.double()
        post_spikes = post_spikes.double()
        # pdb.set_trace()
        # The cases when the input neuron spikes before the latent neuron
        post_pre_LTP = (
            self.trace_pre_ltp[:, t, :].double()  # LTP condition
        ).unsqueeze(-1) @ post_spikes.unsqueeze(1)  # (Batch, in_features, out_features)
        post_pre_LTD = (
            -self.trace_pre_ltd[:, t, :].double()  # LTD condition
        ).unsqueeze(-1) @ post_spikes.unsqueeze(1)  # (Batch, in_features, out_features)

        # The case when the input neuron spikes after the latent neuron
        pre_post = pre_spikes.unsqueeze(-1) @ (
            self.trace_post[:, t, :]  # LTD condition
        ).double().unsqueeze(1)  # (Batch, in_features, out_features)

        dw = (
            (
                (5 * (-self.log_likelihood).exp() - 1) * post_pre_LTP.mean(dim=0)
                - post_pre_LTD.mean(dim=0)
                - pre_post.mean(dim=0)
            )
            * self.learning_rate
            / self.inverse_lr_decay
        )  # (in_features, out_features)
        self.log_likelihood += dw

        db = (
            (
                (5 * (-self.log_prior).exp() - 1) * post_spikes.mean(dim=0)
                - (1 - post_spikes.mean(dim=0))
            )
            * self.learning_rate
            / self.inverse_lr_decay
        )
        self.log_prior += db

        wandb.log(
            {
                "dw": dw.norm().item(),
                "db": db.norm().item(),
            }
        )

        self.inverse_lr_decay += 1
        self.normalize_probs()

    def normalize_probs(self) -> None:
        """Normalize over in_features, to satisfy the constraint that the sum of the probs to each output neuron is 1.
        (the weights is log prob of the input spike given the latent variable)"""
        self.log_likelihood.clamp_(min=-5, max=0)
        population_form = self.log_likelihood.view(
            self.populations, -1, self.out_features
        )
        self.log_likelihood -= (
            population_form.logsumexp(dim=0, keepdim=True)
            .view(-1, self.out_features)
            .repeat(2, 1)
        )
        self.log_prior.clamp_(min=-5, max=0)
        self.log_prior -= self.log_prior.logsumexp(dim=0, keepdim=True)
