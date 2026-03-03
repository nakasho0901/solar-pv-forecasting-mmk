import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    '''
    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves.
    '''
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    '''
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    # CPUの場合は gelsy, GPUの場合は gels を使用
    driver = 'gelsy' if device == 'cpu' else 'gels'
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver=driver).solution[:, :, 0]
    return coef.to(device)


class KANLayer(nn.Module):
    """
    KANLayer class
    """

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-4, 4], sp_trainable=True, sb_trainable=True,
                 device='cpu'): # 修正: デフォルトを 'cpu' に変更
        super(KANLayer, self).__init__()
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device),
                                 torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        noises = noises.to(device)
        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))
        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(
                sb_trainable)
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(
            sp_trainable)
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)
        self.device = device

    def forward(self, x):
        batch = x.shape[0]
        # deviceを現在のxのデバイスに合わせる安全策
        if x.device != self.device:
            self.device = x.device
            
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch,
                                                                                               self.size).permute(1, 0)
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(1, 0)
        y = coef2curve(x_eval=x, grid=self.grid[self.weight_sharing], coef=self.coef[self.weight_sharing], k=self.k,
                       device=self.device)
        y = y.permute(1, 0)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)
        return y

    def update_grid_from_samples(self, x):
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(
            1, 0)
        x_pos = torch.sort(x, dim=1)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k, device=self.device)
        num_interval = self.grid.shape[1] - 1
        ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
        grid_adaptive = x_pos[:, ids]
        margin = 0.01
        grid_uniform = torch.cat(
            [grid_adaptive[:, [0]] - margin + (grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2 * margin) * a for a in
             np.linspace(0, 1, num=self.grid.shape[1])], dim=1)
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)

    def initialize_grid_from_parent(self, parent, x):
        batch = x.shape[0]
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch,
                                                                                                  self.size).permute(1,
                                                                                                                     0)
        x_pos = parent.grid
        sp2 = KANLayer(in_dim=1, out_dim=self.size, k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1))[0].permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)

    def get_subset(self, in_id, out_id):
        spb = KANLayer(len(in_id), len(out_id), self.num, self.k, base_fun=self.base_fun, device=self.device)
        spb.grid.data = self.grid.reshape(self.out_dim, self.in_dim, spb.num + 1)[out_id][:, in_id].reshape(-1,
                                                                                                            spb.num + 1)
        spb.coef.data = self.coef.reshape(self.out_dim, self.in_dim, spb.coef.shape[1])[out_id][:, in_id].reshape(-1,
                                                                                                                  spb.coef.shape[
                                                                                                                      1])
        spb.scale_base.data = self.scale_base.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.scale_sp.data = self.scale_sp.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )
        spb.mask.data = self.mask.reshape(self.out_dim, self.in_dim)[out_id][:, in_id].reshape(-1, )

        spb.in_dim = len(in_id)
        spb.out_dim = len(out_id)
        spb.size = spb.in_dim * spb.out_dim
        return spb

    def lock(self, ids):
        self.lock_counter += 1
        for i in range(len(ids)):
            if i != 0:
                self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[0][1] * self.in_dim + ids[0][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = self.lock_counter

    def unlock(self, ids):
        num = len(ids)
        locked = True
        for i in range(num):
            locked *= (self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] == self.weight_sharing[
                ids[0][1] * self.in_dim + ids[0][0]])
        if locked == False:
            print("they are not locked. unlock failed.")
            return 0
        for i in range(len(ids)):
            self.weight_sharing[ids[i][1] * self.in_dim + ids[i][0]] = ids[i][1] * self.in_dim + ids[i][0]
            self.lock_id[ids[i][1] * self.in_dim + ids[i][0]] = 0
        self.lock_counter -= 1


class WaveKANLayer(nn.Module):
    '''
    Wav-KAN: Wavelet Kolmogorov-Arnold Networks
    '''

    def __init__(self, in_features, out_features, wavelet_type='mexican_hat', with_bn=True, device="cpu"): # 修正: デフォルトを 'cpu' に
        super(WaveKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.with_bn = with_bn
        self.device = device # 追加

        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        self.base_activation = nn.SiLU()

        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'dog':
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi
            def meyer_aux(v):
                return torch.where(v <= 1 / 2, torch.ones_like(v),
                                   torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))))
            def nu(t):
                return t ** 4 * (35 - 84 * t + 70 * t ** 2 - 20 * t ** 3)
            wavelet = torch.sin(pi * v) * meyer_aux(v)
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(x_scaled.size(-1), periodic=False, dtype=x_scaled.dtype,
                                          device=x_scaled.device)
            wavelet = sinc * window
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError("Unsupported wavelet type")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output
        if self.with_bn:
            return self.bn(combined_output)
        else:
            return combined_output


class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300):
        super(NaiveFourierKANLayer, self).__init__()
        self.gridsize = gridsize
        self.inputdim = inputdim
        self.outdim = outdim

        self.fouriercoeffs = nn.Parameter(torch.randn(2, outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = x.view(-1, self.inputdim)
        k = torch.reshape(torch.arange(1, self.gridsize + 1, device=x.device), (1, 1, 1, self.gridsize))
        xrshp = x.view(x.shape[0], 1, x.shape[1], 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)
        c = torch.reshape(c, (1, x.shape[0], x.shape[1], self.gridsize))
        s = torch.reshape(s, (1, x.shape[0], x.shape[1], self.gridsize))
        y = torch.einsum("dbik,djik->bj", torch.concat([c, s], dim=0), self.fouriercoeffs)
        y = y.view(outshape)
        return y


class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree
        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))
        x = torch.tanh(x)
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)
        y = y.view(-1, self.outdim)
        return y


class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        x = torch.tanh(x)
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        x = x.acos()
        x *= self.arange
        x = x.cos()
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        return y


class TaylorKANLayer(nn.Module):
    def __init__(self, input_dim, out_dim, order, addbias=True):
        super(TaylorKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.order = order
        self.addbias = addbias
        self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, order) * 0.01)
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        shape = x.shape
        outshape = shape[0:-1] + (self.out_dim,)
        x = torch.reshape(x, (-1, self.input_dim))
        x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)
        y = torch.zeros((x.shape[0], self.out_dim), device=x.device)
        for i in range(self.order):
            term = (x_expanded ** i) * self.coeffs[:, :, i]
            y += term.sum(dim=-1)
        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y


class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha
        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        nn.init.xavier_uniform_(self.centers)
        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        nn.init.xavier_uniform_(self.weights)

    def gaussian_rbf(self, distances):
        return torch.exp(-self.alpha * distances ** 2)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        basis_values = self.gaussian_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output


class KANInterface(nn.Module):
    def __init__(self, in_features, out_features, layer_type, n_grid=None, degree=None, order=None, n_center=None):
        super(KANInterface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if layer_type == "WavKAN":
            self.transform = WaveKANLayer(in_features, out_features)
        elif layer_type == "KAN":
            self.transform = KANLayer(in_features, out_features, num=n_grid)
        elif layer_type == "FourierKAN":
            self.transform = NaiveFourierKANLayer(in_features, out_features, gridsize=n_grid)
        elif layer_type == "JacobiKAN":
            self.transform = JacobiKANLayer(in_features, out_features, degree=degree)
        elif layer_type == "ChebyKAN":
            self.transform = ChebyKANLayer(in_features, out_features, degree=degree)
        elif layer_type == "TaylorKAN":
            self.transform = TaylorKANLayer(in_features, out_features, order=order)
        elif layer_type == "RBFKAN":
            self.transform = RBFKANLayer(in_features, out_features, num_centers=n_center)
        elif layer_type == "Linear":
            self.transform = nn.Linear(in_features, out_features, bias=True)
        else:
            raise NotImplementedError(f"Layer type {layer_type} not implemented")

    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        return self.transform(x).reshape(B, N, self.out_features)


class KANInterfaceV2(nn.Module):
    def __init__(self, in_features, out_features, layer_type, hyperparam):
        super(KANInterfaceV2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if layer_type == "WavKAN":
            self.transform = WaveKANLayer(in_features, out_features, with_bn=False)
        elif layer_type == "KAN" or layer_type == "SplineKAN": # SplineKANもKANとして扱う
            self.transform = KANLayer(in_features, out_features, num=hyperparam)
        elif layer_type == "FourierKAN":
            self.transform = NaiveFourierKANLayer(in_features, out_features, gridsize=hyperparam)
        elif layer_type == "JacobiKAN":
            self.transform = JacobiKANLayer(in_features, out_features, degree=hyperparam)
        elif layer_type == "ChebyKAN":
            self.transform = ChebyKANLayer(in_features, out_features, degree=hyperparam)
        elif layer_type == "TaylorKAN":
            self.transform = TaylorKANLayer(in_features, out_features, order=hyperparam)
        else:
            raise NotImplementedError(f"Layer type {layer_type} not implemented")

    def forward(self, x):
        x = self.transform(x)
        return x


class MoKLayer(nn.Module):
    def __init__(self, in_features, out_features, experts_type="A", gate_type="Linear"):
        super(MoKLayer, self).__init__()
        # 修正: device='cuda' を削除または 'cpu' に変更
        if experts_type == "A":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                WaveKANLayer(in_features, out_features, wavelet_type="mexican_hat", device="cpu"), # 修正
                WaveKANLayer(in_features, out_features, wavelet_type="mexican_hat", device="cpu")  # 修正
            ])
        elif experts_type == "B":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                JacobiKANLayer(in_features, out_features, degree=6),
                JacobiKANLayer(in_features, out_features, degree=6),
            ])
        elif experts_type == "C":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                TaylorKANLayer(in_features, out_features, order=4, addbias=True),
                JacobiKANLayer(in_features, out_features, degree=5),
                JacobiKANLayer(in_features, out_features, degree=6),
            ])
        elif experts_type == "L":
            self.experts = nn.ModuleList([
                nn.Linear(in_features, out_features),
                nn.Linear(in_features, out_features),
                nn.Linear(in_features, out_features),
                nn.Linear(in_features, out_features),
            ])
        elif experts_type == "V":
            self.experts = nn.ModuleList([
                TaylorKANLayer(in_features, out_features, order=3, addbias=True),
                JacobiKANLayer(in_features, out_features, degree=6),
                WaveKANLayer(in_features, out_features, wavelet_type="mexican_hat", device="cpu"), # 修正
                nn.Linear(in_features, out_features),
            ])
        else:
            raise NotImplemented

        self.n_expert = len(self.experts)
        self.softmax = nn.Softmax(dim=-1)

        if gate_type == "Linear":
            self.gate = nn.Linear(in_features, self.n_expert)
        elif gate_type == "KAN":
            self.gate = JacobiKANLayer(in_features, self.n_expert, degree=5)
        else:
            raise NotImplemented

    def forward(self, x):
        B, N, L = x.shape
        x = x.reshape(B * N, L)
        score = F.softmax(self.gate(x), dim=-1)  # (BxN, E)
        expert_outputs = torch.stack([self.experts[i](x) for i in range(self.n_expert)], dim=-1)  # (BxN, Lo, E)
        return torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, N, -1).contiguous()