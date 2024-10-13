import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn.init import calculate_gain


def xavier_uniform_(tensor, gain=1.):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-a, a)


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(-2)
    num_output_fmaps = tensor.size(-1)
    receptive_field_size = 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        xavier_uniform_(self.weight, gain=1.)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Actor(nn.Module):  # SAC actor输出均值和方差
    def __init__(self, state_dim, action_dim, min_log_std=-10, max_log_std=2):
        # 用于防止出现极端std值
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + 1, 384)
        self.l2 = nn.Linear(384, 256)

        self.mu_head = nn.Linear(256, action_dim)  # 输出均值
        self.log_std_head = nn.Linear(256, action_dim)  # 输出标准差

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, state, uncertain, SR=False):
        su = torch.cat([state, uncertain.log()], -1)
        a = F.relu(self.l1(su))
        a = F.relu(self.l2(a))
        mu = self.mu_head(a)
        if SR:
            return mu, a

        log_std_head = self.log_std_head(a)
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)
        # actor log stddev bounds
        # 这里std需要根据最大动作缩放

        return mu, log_std_head.exp()

    def reset_parameters(self):
        self.l1.reset_parameters()
        self.l2.reset_parameters()

        self.mu_head.reset_parameters()
        self.log_std_head.reset_parameters()
