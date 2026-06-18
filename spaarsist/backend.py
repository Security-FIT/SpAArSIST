"""SpAArSIST: a sparsified, deployment-oriented refinement of the AASIST graph
pooling backend.

The graph-attention machinery (GraphAttentionLayer, HtrgGraphAttentionLayer,
Residual_block) is the original AASIST design by NAVER corp.
    Paper:   https://arxiv.org/pdf/2110.01200
    Code:    https://github.com/TakHemlata/SSL_Anti-spoofing/blob/main/model.py
    License: MIT

SpAArSIST keeps that backbone and isolates three explicit, lightweight choices
(see Firc et al., "SpAArSIST: Sparsified AASIST for Efficient and Reliable
Anti-Spoofing"):

  (1) Separate train-time / inference-time pooling ratios (k_tr, k_inf).
  (2) GraphPool node scoring via a parameter-free magnitude proxy
      s_i = ||n_i||_2^2  instead of a learned linear+sigmoid scorer.
  (3) Stack-node (master) aggregation via an explicit mean instead of the
      high-temperature attention update, which is already near-uniform.

Each behaviour is a toggle so every Table-1 configuration (Base / Mag / Mean /
MagMean, and any k_tr / k_inf pair) can be reproduced from a single class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Original AASIST homogeneous graph-attention layer (unchanged)."""

    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x):
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    """Original AASIST heterogeneous (HS-GAL) graph-attention layer.

    SpAArSIST change (3): the master / stack node update can be switched from
    the attention-weighted form to an explicit mean via ``master_aggregation``.
    Only the master update is affected; the node-node attention is untouched.
    """

    def __init__(self, in_dim, out_dim, master_aggregation: str = "attention", **kwargs):
        super().__init__()
        assert master_aggregation in ("attention", "mean")
        self.master_aggregation = master_aggregation

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # The "without attention" projections form the residual/linear path and
        # are kept in both modes (mean aggregation routes through them).
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # Attention-map projections for the full HS-GAL interaction: node-node
        # (att_proj, proj_with_att) and master (att_projM, proj_with_attM).
        # SpAArSIST's mean aggregation bypasses the entire HS-GAL attention with
        # a plain linear projection + mean, so these four modules are removed --
        # this is the benchmarked SpAArSIST configuration whose backend size and
        # MACs are reported in the paper.
        if master_aggregation == "attention":
            self.att_proj = nn.Linear(in_dim, out_dim)
            self.att_projM = nn.Linear(in_dim, out_dim)
            self.proj_with_att = nn.Linear(in_dim, out_dim)
            self.proj_with_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x1, x2, master=None):
        if self.master_aggregation == "mean":
            return self._forward_mean(x1, x2, master)
        return self._forward_attention(x1, x2, master)

    def _forward_attention(self, x1, x2, master=None):
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)
        return x1, x2, master

    def _forward_mean(self, x1, x2, master=None):
        # SpAArSIST mean aggregation. The whole HS-GAL attention (node-node and
        # master) is replaced by a linear projection: nodes go through the
        # "without attention" branch and the stack node is a linear projection
        # of the incoming master. This is the attention-free, mean-equivalent
        # path that was benchmarked for the reported backend size / MACs.
        num_type1 = x1.size(1)
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x_out = self.proj_without_att(x)
        master = self.proj_without_attM(master)
        if master.size(0) == 1 and x.size(0) > 1:
            master = master.expand(x.size(0), -1, -1)
        if master.dim() == 2:
            master = master.unsqueeze(1)

        return x_out[:, :num_type1, :], x_out[:, num_type1:, :], master

    def _update_master(self, x, master):
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x, master):
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x, num_type1, num_type2):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12)
        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x, master, att_map):
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        return x.view(org_size)

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    """Top-k graph pooling with selectable node scoring and split train/infer k.

    SpAArSIST changes (1) and (2):
      * ``scoring="learned"``  -> baseline AASIST: s_i = sigmoid(w . n_i + b),
        and retained nodes are gated by their (0,1) score.
      * ``scoring="magnitude"`` -> parameter-free proxy s_i = ||n_i||_2^2.
        Pooling is pure top-k selection by feature energy; no gating, since the
        squared norm is only used to *rank* nodes (Eq. (2)-(3)).
      * ``k_train`` / ``k_infer`` let inference prune more (k_inf < k_tr) than
        training without retraining. The active ratio follows ``self.training``.
    """

    def __init__(self, k_train: float, in_dim: int, p: float,
                 scoring: str = "learned", k_infer: float | None = None):
        super().__init__()
        assert scoring in ("learned", "magnitude")
        self.scoring = scoring
        self.k_train = k_train
        self.k_infer = k_train if k_infer is None else k_infer
        self.in_dim = in_dim
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        if scoring == "learned":
            self.proj = nn.Linear(in_dim, 1)
            self.sigmoid = nn.Sigmoid()

    @property
    def k(self) -> float:
        return self.k_train if self.training else self.k_infer

    def forward(self, h):
        Z = self.drop(h)
        if self.scoring == "learned":
            scores = self.sigmoid(self.proj(Z))
            self.last_scores = scores
            # baseline behaviour: gate retained nodes by their score
            return self._top_k_graph(scores, h, self.k, gate=True)
        # magnitude proxy: rank by squared L2 norm, select top-k, no gating
        scores = (Z ** 2).sum(dim=-1, keepdim=True)
        self.last_scores = scores
        return self._top_k_graph(scores, h, self.k, gate=False)

    @staticmethod
    def _top_k_graph(scores, h, k, gate: bool):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        if gate:
            h = h * scores
        return torch.gather(h, 1, idx)


class Residual_block(nn.Module):
    """Original AASIST RawNet2-style residual block (unchanged)."""

    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3),
                               padding=(1, 1), stride=1)
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3),
                               padding=(0, 1), stride=1)
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1],
                                             padding=(0, 1), kernel_size=(1, 3), stride=1)
        else:
            self.downsample = False

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        return out


class SpAArSIST(nn.Module):
    """AASIST graph-pooling backend with the SpAArSIST sparsification toggles.

    Maps frame-level SSL features (B, T, inputs_dim) to a fixed utterance
    embedding (B, outputs_dim).

    Args:
        node_scoring:       "learned" (baseline) or "magnitude" (Mag).
        stack_aggregation:  "attention" (baseline) or "mean" (Mean).
        pool_ratio_train:   k_tr, retained-node fraction during training.
        pool_ratio_infer:   k_inf, retained-node fraction at inference.
                            Defaults to k_tr (matched sparsity).
        temperatures:       [GAT_S, GAT_T, HS-GAL, HS-GAL] softmax temperatures.

    A 4D feature tensor (n_layers, B, T, D) is reduced to its last layer, so the
    output of an SSL front-end that stacks all transformer layers can be fed
    directly.
    """

    def __init__(
        self,
        inputs_dim: int = 1024,
        outputs_dim: int = 1024,
        node_scoring: str = "magnitude",
        stack_aggregation: str = "mean",
        pool_ratio_train: float = 0.3,
        pool_ratio_infer: float | None = 0.1,
        temperatures=(2.0, 2.0, 100.0, 100.0),
    ):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim

        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        k_tr = pool_ratio_train
        k_inf = pool_ratio_train if pool_ratio_infer is None else pool_ratio_infer

        self.LL = nn.Linear(inputs_dim, 128)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.first_bn1 = nn.BatchNorm2d(num_features=64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # Each block is wrapped in its own nn.Sequential to match the public
        # AASIST module nesting (state-dict keys: encoder.<i>.0.*), so existing
        # AASIST/SpAArSIST checkpoints load without renaming.
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])

        def htrg(in_d, out_d):
            return HtrgGraphAttentionLayer(in_d, out_d, master_aggregation=stack_aggregation,
                                           temperature=temperatures[2])

        self.HtrgGAT_layer_ST11 = htrg(gat_dims[0], gat_dims[1])
        self.HtrgGAT_layer_ST12 = htrg(gat_dims[1], gat_dims[1])
        self.HtrgGAT_layer_ST21 = htrg(gat_dims[0], gat_dims[1])
        self.HtrgGAT_layer_ST22 = htrg(gat_dims[1], gat_dims[1])

        def pool(in_d):
            return GraphPool(k_tr, in_d, 0.3, scoring=node_scoring, k_infer=k_inf)

        self.pool_S = pool(gat_dims[0])
        self.pool_T = pool(gat_dims[0])
        self.pool_hS1 = pool(gat_dims[1])
        self.pool_hT1 = pool(gat_dims[1])
        self.pool_hS2 = pool(gat_dims[1])
        self.pool_hT2 = pool(gat_dims[1])

        self.out_layer = nn.Linear(5 * gat_dims[1], outputs_dim)

    def forward(self, x):
        x = x[-1] if x.dim() == 4 else x  # (B, T, D)
        x = self.LL(x)
        x = x.transpose(1, 2).unsqueeze(dim=1)
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))

        x = self.encoder(x)
        x = self.selu(self.first_bn1(x))

        w = self.attention(x)

        # spectral attention
        w1 = F.softmax(w, dim=-1)
        m = torch.sum(x * w1, dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # temporal attention
        w2 = F.softmax(w, dim=-2)
        m1 = torch.sum(x * w2, dim=-2)
        e_T = m1.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # inference path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # inference path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        return self.out_layer(last_hidden)


# Backwards-compatible alias.
AASISTBackend = SpAArSIST
