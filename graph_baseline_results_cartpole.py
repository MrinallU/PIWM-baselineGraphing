import torch
from torch.utils.data import Dataset, DataLoader
import math, os, time, random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


GOKU_CARTPOLE_WEIGHTS_PATH = "./goku_cartpole_weights.pth"
VID2PARA_CARTPOLE_WEIGHTS_PATH = "./v2p_cartpole_weights.pth"
DVBF_CARTPOLE_WEIGHTS_PATH = "./dvfb_cartpole_weights.pth"
SINDYC_CARTPOLE_WEIGHTS_PATH = "./sindyc_cartpole_weights.pth"


# ========= Configuration =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAU = 0.02  # Discrete time step
PRED_STEPS = 30  # Number of prediction steps for testing
IMG_H, IMG_W = 120, 160
IMG_C = 3
LATENT_Z_DIM = 4  # Physical state z=[x,y,theta,v]
ACT_DIM = 2  # action=[steer, accel]
ENC_HIDDEN = 128
RNN_HIDDEN = 64
KL_INIT = 1e-5  # KL annealing starting point
KL_FINAL = 1.0
ANNEAL_EPOCHS = 20  # KL annealing epochs
LR = 1e-3


# ========= Model Components =========
class ImageEncoder(nn.Module):
    """Encode each frame (C,H,W) into frame features"""

    def __init__(self, out_dim=ENC_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(IMG_C, 32, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):  # (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        f = self.net(x).flatten(1)
        f = self.proj(f)  # (B*T, out_dim)
        return f.view(B, T, -1)  # (B,T,out_dim)


class Z0Encoder(nn.Module):
    """Encode initial z0 (mu, logvar)"""

    def __init__(self, in_dim=ENC_HIDDEN, rnn_hidden=RNN_HIDDEN, z_dim=LATENT_Z_DIM):
        super().__init__()
        self.rnn = nn.GRU(
            in_dim, rnn_hidden, batch_first=True
        )  # Unidirectional, read temporally
        self.mu = nn.Linear(rnn_hidden, z_dim)
        self.logvar = nn.Linear(rnn_hidden, z_dim)

    def forward(self, feat):  # (B,T,D)
        _, h = self.rnn(feat)  # h:(1,B,H)
        h = h[0]
        return self.mu(h), self.logvar(h)


class ThetaEncoder(nn.Module):
    """Encode static parameters (wheelbase L) (mu, logvar)"""

    def __init__(self, in_dim=ENC_HIDDEN, rnn_hidden=RNN_HIDDEN):
        super().__init__()
        self.bilstm = nn.LSTM(in_dim, rnn_hidden, bidirectional=True, batch_first=True)
        self.mu = nn.Linear(2 * rnn_hidden, 1)
        self.logvar = nn.Linear(2 * rnn_hidden, 1)

    def forward(self, feat):  # (B,T,D)
        out, _ = self.bilstm(feat)
        h = out.mean(dim=1)  # Pooling
        return self.mu(h), self.logvar(h)


class H_z(nn.Module):
    """Map standard Gaussian space z0 to physical space"""

    def __init__(self, z_dim=LATENT_Z_DIM):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, z_dim))

    def forward(self, z_tilde):
        # x,y,theta,v without singular constraints; theta can add range compression (optional)
        return self.mlp(z_tilde)


class H_theta(nn.Module):
    """Map standard Gaussian theta to physically feasible wheelbase L>0"""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
        self.softplus = nn.Softplus()

    def forward(self, theta_tilde):
        # L = softplus(MLP) + L_min, avoid being too small
        L_min = 0.5  # You can set reasonable lower bound according to data (same unit as state)
        return self.softplus(self.mlp(theta_tilde)) + L_min


class BicycleDynamics(nn.Module):
    """Known bicycle model one-step update, L output by network"""

    def __init__(self, tau=TAU):
        super().__init__()
        self.tau = tau

    def forward(self, z, a, L):
        # z: (B,4) = [x,y,theta,v]; a: (B,2) = [delta, accel]; L: (B,1)
        x, y, theta, v = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        delta = a[:, 0]
        accel = a[:, 1]
        x_next = x + v * torch.cos(theta) * self.tau
        y_next = y + v * torch.sin(theta) * self.tau
        theta_next = theta + (v / (L.squeeze(1))) * torch.tan(delta) * self.tau
        v_next = v + accel * self.tau
        return torch.stack([x_next, y_next, theta_next, v_next], dim=1)


class ImageDecoder(nn.Module):
    """Decode physical state z_t back to image"""

    def __init__(self, in_dim=LATENT_Z_DIM, base=ENC_HIDDEN):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, base * 6 * 8), nn.ReLU())
        # 96×128 is also fine, here directly upsample to 120×160
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base, 128, 4, 2, 1),
            nn.ReLU(),  # 6x8 -> 12x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),  # 12x16 -> 24x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),  # 24x32 -> 48x64
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ReLU(),  # 48x64 -> 96x128
            nn.Conv2d(32, IMG_C, 3, 1, 1),
        )

    def forward(self, z):  # (B,T,4)
        B, T, _ = z.shape
        h = self.fc(z.reshape(B * T, -1))  # (B*T, base*6*8)
        h = h.view(B * T, ENC_HIDDEN, 6, 8)  # (B*T, C, 6, 8)
        img = self.deconv(h)  # (B*T, C, 96, 128)
        # Simple resize to 120x160 (maintain end-to-end, avoid interpolation operator non-differentiable issues, use Conv instead)
        pad_h = 120 - img.shape[2]
        pad_w = 160 - img.shape[3]
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
        return img.view(B, T, IMG_C, 120, 160)


# ========= Main Model (End-to-End) =========
class GOKU_Bicycle(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_img = ImageEncoder()
        self.enc_z0 = Z0Encoder()
        self.enc_th = ThetaEncoder()
        self.hz = H_z()
        self.htheta = H_theta()
        self.dyn = BicycleDynamics()
        self.dec_img = ImageDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def run_cartpole(self):
        ckpt = torch.load(GOKU_CARTPOLE_WEIGHTS_PATH, map_location="cpu")
        rmse_curve = ckpt["rmse_curve"]

        rmse_vals = rmse_curve.detach().cpu().numpy()[:30]
        steps = np.arange(1, len(rmse_vals) + 1, dtype=np.int32)
        return steps, rmse_vals

    def forward(self, x, action, predict_steps=0):
        """
        x: (B,T,C,H,W), action: (B,T,2)
        Return: reconstructed frames (including prediction), z sequence, L
        """
        B, T, C, H, W = x.shape
        feat = self.enc_img(x)  # (B,T,D)

        # Encode z0 and wheelbase L
        mu_z0, logvar_z0 = self.enc_z0(feat)
        mu_th, logvar_th = self.enc_th(feat)

        z0_tilde = self.reparameterize(mu_z0, logvar_z0)  # (B,4)
        th_tilde = self.reparameterize(mu_th, logvar_th)  # (B,1)

        z0 = self.hz(z0_tilde)  # Physical initial state
        L = self.htheta(th_tilde)  # Physical wheelbase >0

        # Use known dynamics to advance
        T_total = T + predict_steps
        z_list = [z0]
        # Action sequence: if prediction steps needed, repeat last action (or set to zero)
        if predict_steps > 0:
            a_pad = torch.cat(
                [action, action[:, -1:].expand(B, predict_steps, ACT_DIM)], dim=1
            )
        else:
            a_pad = action
        for t in range(T_total - 1):
            z_next = self.dyn(z_list[-1], a_pad[:, t, :], L)
            z_list.append(z_next)
        z_seq = torch.stack(z_list, dim=1)  # (B, T_total, 4)

        # Decode
        x_hat = self.dec_img(z_seq)  # (B, T_total, C, H, W)

        # KL
        kl_z0 = (
            -0.5
            * torch.sum(1 + logvar_z0 - mu_z0.pow(2) - logvar_z0.exp(), dim=1).mean()
        )
        kl_th = (
            -0.5
            * torch.sum(1 + logvar_th - mu_th.pow(2) - logvar_th.exp(), dim=1).mean()
        )

        return x_hat, z_seq, L, kl_z0, kl_th, mu_z0, mu_th


# --- 1. 模块定义 ---
# 将模型中的各个组件封装为独立的 PyTorch 模块，提高代码可读性。


class Encoder(nn.Module):
    """
    VRNN编码器：将图像和状态信息编码成潜在表示。
    输入: 图像 (B, C, H, W) 和状态 (B, state_dim)
    输出: 潜在空间的均值 (mu) 和对数方差 (logvar)
    """

    def __init__(self, image_channels, state_dim, hidden_dim, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_img = nn.Linear(64 * 16 * 16, hidden_dim)  # 假设输入图像为 64x64
        self.fc_state = nn.Linear(state_dim, hidden_dim)
        self.fc_merge = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, image, state):
        img_features = self.fc_img(self.conv(image))
        state_features = self.fc_state(state)
        merged_features = self.fc_merge(
            torch.cat([img_features, state_features], dim=1)
        )
        mu = self.fc_mu(merged_features)
        logvar = self.fc_logvar(merged_features)
        return mu, logvar


class Decoder(nn.Module):
    """
    VRNN解码器：将潜在表示和RNN隐藏状态解码回图像。
    输入: 潜在表示 (z) 和隐藏状态 (rnn_h)
    输出: 重建图像
    """

    def __init__(self, z_dim, hidden_dim, image_channels):
        super().__init__()
        self.fc = nn.Linear(z_dim + hidden_dim, 64 * 16 * 16)
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, rnn_h):
        z_combined = torch.cat([z, rnn_h], dim=1)
        x = self.fc(z_combined).view(-1, 64, 16, 16)
        return self.conv_trans(x)


class Prior(nn.Module):
    """
    先验网络：基于RNN的隐藏状态预测潜在表示的先验分布。
    输入: RNN隐藏状态 (rnn_h)
    输出: 先验分布的均值 (mu) 和对数方差 (logvar)
    """

    def __init__(self, hidden_dim, z_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, rnn_h):
        mu = self.fc_mu(rnn_h)
        logvar = self.fc_logvar(rnn_h)
        return mu, logvar


class VRNNCar(nn.Module):
    """
    完整的VRNN模型，用于车辆动力学参数推断。
    它包含编码器、解码器、RNN单元和物理参数预测层。
    """

    def __init__(self, image_channels, state_dim, hidden_dim, z_dim, l_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.l_dim = l_dim

        self.encoder = Encoder(image_channels, state_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, image_channels)
        self.rnn = nn.GRU(z_dim, hidden_dim)  # 使用GRU作为循环单元
        self.prior = Prior(hidden_dim, z_dim)

        # 物理参数 L (轴距) 的映射层
        self.fc_l_mu = nn.Linear(z_dim, l_dim)
        self.fc_l_logvar = nn.Linear(z_dim, l_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧，用于从潜在分布中采样。"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def run_cartpole(self):
        ckpt = torch.load(VID2PARA_CARTPOLE_WEIGHTS_PATH, map_location="cpu")
        rmse_curve = ckpt["rmse_curve"]

        rmse_vals = rmse_curve.detach().cpu().numpy()[:30]
        steps = np.arange(1, len(rmse_vals) + 1, dtype=np.int32)
        return steps, rmse_vals

    def forward(self, x_seq, s_seq, h_0=None):
        """
        前向传播：处理一个完整的序列。
        输入:
            x_seq (T, B, C, H, W): 图像序列
            s_seq (T, B, S_dim): 状态序列
            h_0 (1, B, H_dim): 初始RNN隐藏状态
        输出:
            x_rec_seq (T, B, C, H, W): 重建图像序列
            mu_l_seq, logvar_l_seq (T, B, 1): 轴距L的预测分布
            mu_enc_seq, logvar_enc_seq (T, B, z_dim): 编码器潜在空间的分布
            mu_prior_seq, logvar_prior_seq (T, B, z_dim): 先验分布
        """
        T, B, _, _, _ = x_seq.size()

        # 初始化存储列表
        x_rec_seq, mu_l_seq, logvar_l_seq = [], [], []
        mu_enc_seq, logvar_enc_seq = [], []
        mu_prior_seq, logvar_prior_seq = [], []

        # 初始化RNN隐藏状态
        h_t = (
            h_0
            if h_0 is not None
            else torch.zeros(1, B, self.hidden_dim, device=x_seq.device)
        )

        for t in range(T):
            # 获取当前时间步的数据
            x_t = x_seq[t]
            s_t = s_seq[t]

            # 1. 先验预测：基于上一步的隐藏状态预测当前潜在空间的先验分布
            mu_prior_t, logvar_prior_t = self.prior(h_t.squeeze(0))

            # 2. 编码器推断：基于当前图像和状态推断潜在分布
            mu_enc_t, logvar_enc_t = self.encoder(x_t, s_t)
            z_t = self.reparameterize(mu_enc_t, logvar_enc_t)

            # 3. 物理参数预测：从潜在表示中推断轴距 L
            mu_l_t = self.fc_l_mu(z_t)
            logvar_l_t = self.fc_l_logvar(z_t)

            # 4. 解码器生成：从潜在表示和隐藏状态重建图像
            x_rec_t = self.decoder(z_t, h_t.squeeze(0))

            # 5. RNN更新：使用推断出的 z 更新隐藏状态
            _, h_t = self.rnn(z_t.unsqueeze(0), h_t)

            # 存储结果
            x_rec_seq.append(x_rec_t)
            mu_l_seq.append(mu_l_t)
            logvar_l_seq.append(logvar_l_t)
            mu_enc_seq.append(mu_enc_t)
            logvar_enc_seq.append(logvar_enc_t)
            mu_prior_seq.append(mu_prior_t)
            logvar_prior_seq.append(logvar_prior_t)

        return (
            torch.stack(x_rec_seq),
            torch.stack(mu_l_seq),
            torch.stack(logvar_l_seq),
            torch.stack(mu_enc_seq),
            torch.stack(logvar_enc_seq),
            torch.stack(mu_prior_seq),
            torch.stack(logvar_prior_seq),
        )


# =========================
# Dataset Class
# =========================
class TrajectoryDataset(Dataset):
    """轨迹预测数据集"""

    def __init__(self, trajectory_file):
        """
        Args:
            trajectory_file: 轨迹数据文件路径
        """

        data = np.load(trajectory_file)
        self.z_current = data["z_current"]  # (N, 512) - z_t
        self.actions = data["actions"]  # (N, 2) - u_t
        self.z_next = data["z_next"]  # (N, 512) - z_{t+1}

    def __len__(self):
        return len(self.z_current)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.z_current[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.z_next[idx]),
        )


# =========================
# Linear Dynamics Model
# =========================
class LinearDynamicsModel(nn.Module):
    """
    线性动态模型: z_{t+1} = A_t * z_t + B_t * u_t
    A_t, B_t 由神经网络预测得出
    """

    def __init__(self, latent_dim=512, action_dim=2, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # 神经网络预测局部线性矩阵 A_t 和 B_t
        # 输入: z_t 和 u_t 的拼接
        input_dim = latent_dim + action_dim

        # 预测 A_t 矩阵 (latent_dim x latent_dim)
        self.A_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim * latent_dim),
        )

        # 预测 B_t 矩阵 (latent_dim x action_dim)
        self.B_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, latent_dim * action_dim),
        )

        # 初始化权重
        self._initialize_weights()

    def run_cartpole(self):
        ckpt = torch.load(DVBF_CARTPOLE_WEIGHTS_PATH, map_location="cpu")
        rmse_curve = ckpt["rmse_curve"]

        rmse_vals = rmse_curve.detach().cpu().numpy()[:30]
        steps = np.arange(1, len(rmse_vals) + 1, dtype=np.int32)
        return steps, rmse_vals

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # A_network 的最后一层初始化为接近单位矩阵
        with torch.no_grad():
            identity_vec = torch.eye(self.latent_dim).flatten()
            self.A_network[-1].weight.data.fill_(0.01)
            self.A_network[-1].bias.data.copy_(identity_vec)

    def forward(self, z_t, u_t):
        """
        Args:
            z_t: (batch_size, latent_dim) - 当前latent状态
            u_t: (batch_size, action_dim) - 当前动作
        Returns:
            z_next_pred: (batch_size, latent_dim) - 预测的下一个latent状态
        """
        batch_size = z_t.size(0)

        # 拼接输入
        input_features = torch.cat(
            [z_t, u_t], dim=1
        )  # (batch_size, latent_dim + action_dim)

        # 预测矩阵 A_t 和 B_t
        A_flat = self.A_network(input_features)  # (batch_size, latent_dim^2)
        B_flat = self.B_network(input_features)  # (batch_size, latent_dim * action_dim)

        # 重塑为矩阵形式
        A_t = A_flat.view(
            batch_size, self.latent_dim, self.latent_dim
        )  # (batch_size, latent_dim, latent_dim)
        B_t = B_flat.view(
            batch_size, self.latent_dim, self.action_dim
        )  # (batch_size, latent_dim, action_dim)

        # 线性动态方程: z_{t+1} = A_t * z_t + B_t * u_t
        z_t_expanded = z_t.unsqueeze(-1)  # (batch_size, latent_dim, 1)
        u_t_expanded = u_t.unsqueeze(-1)  # (batch_size, action_dim, 1)

        Az_term = torch.bmm(A_t, z_t_expanded).squeeze(-1)  # (batch_size, latent_dim)
        Bu_term = torch.bmm(B_t, u_t_expanded).squeeze(-1)  # (batch_size, latent_dim)

        z_next_pred = Az_term + Bu_term  # (batch_size, latent_dim)

        return z_next_pred


# SINDYc参数
SINDY_LAMBDA = 0.02  # 较小的lambda以发现更丰富的动力学
SINDY_POLY_ORDER = 2
SINDY_MAX_ITER = 10


# =========================
# SINDYc Implementation - 从origin.py提取并简化
# =========================
def build_library(
    x: np.ndarray, u: np.ndarray, include_bias=True, poly_order=2, include_cross=True
) -> np.ndarray:
    """
    构建SINDYc库矩阵 - 针对4维状态优化
    x: (N, 4) - 状态变量 [x, y, theta, speed]
    u: (N, 2) - 控制输入 [steering, throttle]
    """
    X = []

    # 常数项
    if include_bias:
        X.append(np.ones((x.shape[0], 1)))

    # 一次项
    X.append(x)  # 状态项 x, y, theta, speed
    X.append(u)  # 控制项 steering, throttle

    if poly_order >= 2:
        # 二次项
        X.append(x**2)  # x^2, y^2, theta^2, speed^2
        X.append(u**2)  # steering^2, throttle^2

        if include_cross:
            # 状态交叉项 (只包含有意义的组合)
            nx = x.shape[1]
            cross_x = []
            for i in range(nx):
                for j in range(i + 1, nx):
                    cross_x.append((x[:, i] * x[:, j])[:, None])
            if len(cross_x) > 0:
                X.append(np.concatenate(cross_x, axis=1))

            # 状态-控制交叉项
            cross_xu = []
            for i in range(x.shape[1]):
                for j in range(u.shape[1]):
                    cross_xu.append((x[:, i] * u[:, j])[:, None])
            if len(cross_xu) > 0:
                X.append(np.concatenate(cross_xu, axis=1))

    Theta = np.concatenate(X, axis=1)
    return Theta


def stlsq(
    Theta: np.ndarray, Y: np.ndarray, lam: float = 0.05, max_iter: int = 10
) -> np.ndarray:
    """
    Sequentially Thresholded Least Squares
    """
    Xi, _, _, _ = np.linalg.lstsq(Theta, Y, rcond=None)

    for _ in range(max_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0

        for k in range(Y.shape[1]):
            big_idx = ~small[:, k]
            if np.sum(big_idx) == 0:
                continue
            Xi[big_idx, k], _, _, _ = np.linalg.lstsq(
                Theta[:, big_idx], Y[:, k], rcond=None
            )

    return Xi


class SINDYC:
    """SINDYc模型类 - 针对物理状态优化"""

    def __init__(self, lam=0.02, poly_order=2, max_iter=10):
        self.lam = lam
        self.poly_order = poly_order
        self.max_iter = max_iter
        self.Xi = None
        self.feature_names = None
        self.include_bias = True
        self.include_cross = True

    def _make_feature_names(self, nx, nu):
        """生成特征名称 - 使用物理变量名"""
        names = []

        if self.include_bias:
            names.append("1")

        # 状态变量
        state_vars = ["x", "y", "theta", "v"]  # 位置x, 位置y, 航向角theta, 速度v
        for i in range(nx):
            names.append(state_vars[i])

        # 控制变量
        control_vars = ["delta", "a"]  # 转向角delta, 加速度a
        for j in range(nu):
            names.append(control_vars[j])

        if self.poly_order >= 2:
            # 状态二次项
            for i in range(nx):
                names.append(f"{state_vars[i]}^2")

            # 控制二次项
            for j in range(nu):
                names.append(f"{control_vars[j]}^2")

            if self.include_cross:
                # 状态交叉项
                for i in range(nx):
                    for j in range(i + 1, nx):
                        names.append(f"{state_vars[i]}*{state_vars[j]}")

                # 状态-控制交叉项
                for i in range(nx):
                    for j in range(nu):
                        names.append(f"{state_vars[i]}*{control_vars[j]}")

        return names

    def fit(self, x: np.ndarray, u: np.ndarray, y_next: np.ndarray):
        """拟合SINDYc模型"""

        # 构建库矩阵
        Theta = build_library(
            x,
            u,
            include_bias=self.include_bias,
            poly_order=self.poly_order,
            include_cross=self.include_cross,
        )

        # 稀疏回归
        self.Xi = stlsq(Theta, y_next, lam=self.lam, max_iter=self.max_iter)

        # 生成特征名称
        self.feature_names = self._make_feature_names(nx=x.shape[1], nu=u.shape[1])

        # 计算稀疏度
        total_coefs = self.Xi.size
        nonzero_coefs = np.sum(np.abs(self.Xi) > 1e-8)
        sparsity = 1.0 - nonzero_coefs / total_coefs

        return self

    def predict_next(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """预测下一状态"""
        if self.Xi is None:
            raise RuntimeError("Model not fitted yet")

        Theta = build_library(
            x,
            u,
            include_bias=self.include_bias,
            poly_order=self.poly_order,
            include_cross=self.include_cross,
        )
        return Theta @ self.Xi

    def run_cartpole(self):
        ckpt = torch.load(SINDYC_CARTPOLE_WEIGHTS_PATH, map_location="cpu")
        rmse_curve = ckpt["rmse_curve"]

        rmse_vals = rmse_curve.detach().cpu().numpy()[:30]
        steps = np.arange(1, len(rmse_vals) + 1, dtype=np.int32)
        return steps, rmse_vals

    def pretty_print(self, top_k=8):
        """打印发现的动力学方程 - 使用物理变量名"""
        if self.Xi is None:
            return "Model not fitted"

        lines = []
        state_vars = ["x", "y", "theta", "v"]
        ny = self.Xi.shape[1]

        for k in range(ny):
            coefs = self.Xi[:, k]
            nonzero_idx = np.where(np.abs(coefs) > 1e-8)[0]
            sorted_terms = sorted(
                [(i, coefs[i]) for i in nonzero_idx], key=lambda t: -abs(t[1])
            )

            eq = [f"{state_vars[k]}(t+1) = "]

            for i, (idx, coef) in enumerate(sorted_terms[:top_k]):
                if i == 0:
                    sign = "" if coef >= 0 else "-"
                else:
                    sign = " + " if coef >= 0 else " - "

                eq.append(f"{sign}{abs(coef):.4g}*{self.feature_names[idx]}")

            if len(sorted_terms) == 0:
                eq.append("0")

            lines.append("".join(eq))

        return "\n".join(lines)


goku = GOKU_Bicycle()
goku_steps, goku_rmse = goku.run_cartpole()

vid2para = VRNNCar(image_channels=3, state_dim=4, hidden_dim=256, z_dim=128, l_dim=1)
v2p_steps, v2p_rmse = vid2para.run_cartpole()

dvbf = LinearDynamicsModel()
dvbf_steps, dvbf_rmse = dvbf.run_cartpole()

sindyc = SINDYC()
sindyc_steps, sindyc_rmse = sindyc.run_cartpole()

# --- Graph 1: GOKU + Vid2Para ---
plt.figure()
plt.plot(goku_steps, goku_rmse, label="GOKU")
plt.plot(v2p_steps, v2p_rmse, label="Vid2Para")
plt.xlabel("Prediction Horizons")
plt.ylabel("RMSE")
plt.title("Intrinsic")
plt.legend()
plt.show()

# --- Graph 2: DVBF + SINDYc ---
plt.figure()
plt.plot(dvbf_steps, dvbf_rmse, label="DVBF")
plt.plot(sindyc_steps, sindyc_rmse, label="SINDYc")
plt.xlabel("Prediction Horizons")
plt.ylabel("RMSE")
plt.title("Extrinsic")
plt.legend()
plt.show()
