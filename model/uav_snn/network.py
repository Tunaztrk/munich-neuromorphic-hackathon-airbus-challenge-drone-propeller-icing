"""Two recurrent LIF layers and a continuous residual trajectory decoder.

One SNN step represents one actual sampled sensor interval. Binary spikes are
used in the forward pass; a fast-sigmoid surrogate is used only in backward.
This is an original compact prototype, not a reproduction of SeqSNN.
"""
import math

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, slope):
        ctx.save_for_backward(voltage)
        ctx.slope = slope
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        (voltage,) = ctx.saved_tensors
        return gradient/(1+ctx.slope*voltage.abs()).square(), None


class RecurrentLIF(nn.Module):
    def __init__(self, input_size, hidden_size, cfg, time_constant):
        super().__init__()
        self.input = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.orthogonal_(self.recurrent.weight, gain=.1)
        nn.init.constant_(self.input.bias, .1)
        beta = math.exp(-cfg['sample_interval_seconds']/time_constant)
        self.decay_logit = nn.Parameter(torch.full((hidden_size,), math.log(beta/(1-beta))),
                                       requires_grad=cfg['learn_decay'])
        self.threshold = cfg['spike_threshold']
        self.slope = cfg['surrogate_slope']
        self.hidden_size = hidden_size

    def step(self, x, state):
        if state is None:
            membrane = x.new_zeros((len(x), self.hidden_size))
            previous_spikes = torch.zeros_like(membrane)
        else:
            membrane, previous_spikes = state
        voltage = self.decay_logit.sigmoid()*membrane+self.input(x)+self.recurrent(previous_spikes)
        spikes = SurrogateSpike.apply(voltage-self.threshold, self.slope)
        # Subtractive reset; detach the reset path to stabilize surrogate BPTT.
        membrane = voltage-self.threshold*spikes.detach()
        return spikes, (membrane, spikes)


class CurrentSNN(nn.Module):
    def __init__(self, feature_count, horizon, cfg, current_index=0,
                 anchor_scale=1., anchor_shift=0.):
        super().__init__()
        hidden = cfg['hidden_size']
        self.layers = nn.ModuleList([
            RecurrentLIF(feature_count, hidden, cfg, cfg['neuron_time_constants_seconds'][0]),
            RecurrentLIF(hidden, hidden, cfg, cfg['neuron_time_constants_seconds'][1]),
        ])
        self.decoder = nn.Sequential(nn.Linear(2*hidden, hidden), nn.Tanh(), nn.Linear(hidden, horizon))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
        self.trace_decay = math.exp(-cfg['sample_interval_seconds']/cfg['trace_time_constant_seconds'])
        self.current_index = current_index
        self.register_buffer('anchor_scale', torch.tensor(float(anchor_scale)))
        self.register_buffer('anchor_shift', torch.tensor(float(anchor_shift)))

    def encode_step(self, x, state=None):
        """Return the frozen representation immediately before the final head."""
        if state is None:
            states = [None, None]
            trace = x.new_zeros((len(x), self.layers[-1].hidden_size))
        else:
            states, trace = state
        activity = []
        new_states = []
        hidden = x
        for layer, layer_state in zip(self.layers, states):
            hidden, layer_state = layer.step(hidden, layer_state)
            activity.append(hidden.mean(dim=1))
            new_states.append(layer_state)
        trace = self.trace_decay*trace+(1-self.trace_decay)*hidden
        membrane = new_states[-1][0]
        anchor = x[:, self.current_index:self.current_index+1]*self.anchor_scale+self.anchor_shift
        representation = self.decoder[1](self.decoder[0](torch.cat([torch.tanh(membrane), trace], dim=-1)))
        return representation, anchor, (new_states, trace), torch.stack(activity, dim=-1)

    def step(self, x, state=None):
        representation, anchor, state, activity = self.encode_step(x, state)
        return anchor+self.decoder[-1](representation), state, activity

    def window_encode(self, x, lengths=None):
        """Independent causal windows, reset state once per window.

        Right padding is ignored: capture each lane's last *observed* state.
        Overlapping windows never inherit state from the preceding window.
        """
        if lengths is None:
            lengths = torch.full((len(x),), x.shape[1], device=x.device, dtype=torch.long)
        if x.shape[1] < 1 or bool(((lengths < 1) | (lengths > x.shape[1])).any()):
            raise ValueError('Window lengths must be between 1 and the padded width.')
        state = None
        result = x.new_zeros((len(x), self.layers[-1].hidden_size))
        last_anchor = x.new_zeros((len(x), 1))
        activity = x.new_zeros((len(x), 2))
        for k, sample in enumerate(x.unbind(dim=1)):
            z, anchor, state, spikes = self.encode_step(sample, state)
            last = (lengths == k+1).unsqueeze(1)
            result = torch.where(last, z, result)
            last_anchor = torch.where(last, anchor, last_anchor)
            activity = activity+spikes*(lengths > k).unsqueeze(1)
        return result, last_anchor, activity/lengths.unsqueeze(1)

    def forward(self, x, state=None):
        predictions, activity = [], []
        for sample in x.unbind(dim=1):
            prediction, state, spikes = self.step(sample, state)
            predictions.append(prediction)
            activity.append(spikes)
        return torch.stack(predictions, dim=1), state, torch.stack(activity, dim=1)

    @staticmethod
    def detach_state(state):
        states, trace = state
        return [(u.detach(), s.detach()) for u, s in states], trace.detach()


def build_model(data, cfg):
    j = data.current_index
    return CurrentSNN(len(data.feature_names), data.horizon, cfg, j,
                      float(data.scale[j])/data.target_scale,
                      (float(data.center[j])-data.target_mean)/data.target_scale)
