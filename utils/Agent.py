import copy
import torch
import torch.nn.functional as F
from utils.networks import Actor, Critic_Q, Temperature  # 这里是agent的网络
from torch.distributions import Normal, Uniform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
min_Val = torch.tensor(1e-7).float()


class Agent():
    def __init__(
            self,
            state_dim,
            action_dim,
    ):
        # 初始化actor,没有目标策略
        self.actor = Actor(state_dim, action_dim).to(device).share_memory()
        self.one = torch.ones(1, 1, device=device)

    def select_best_action(self, state, uncertain=0.25):  # 使用agent选择动作
        if isinstance(state, torch.Tensor):
            state = state.float().reshape(1, -1).to(device)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        mu, log_sigma = self.actor(state, uncertain * self.one)
        return torch.tanh(mu).cpu().data.numpy().flatten()

    def load(self, filename):  # 加载模型
        # print(torch.load(filename + "_actor_optimizer", map_location=torch.device('cpu')))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=torch.device('cpu')))
