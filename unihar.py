import math

import torch
from torch import nn
from torch.nn import functional as F


SCALES = ("f", "m", "c")


def pool_tokens(x, target_len):
    return F.adaptive_avg_pool1d(x.transpose(1, 2), target_len).transpose(1, 2)


class ChannelDescriptor(nn.Module):
    def __init__(self, d_psi=16, eps=1e-6):
        super().__init__()
        if d_psi < 4:
            raise ValueError("d_psi must be at least 4")
        self.num_spectral = d_psi - 4
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1)
        centered = x - mean.unsqueeze(1)
        std = (centered.square().mean(dim=1) + self.eps).sqrt()
        skew = centered.pow(3).mean(dim=1) / (std.pow(3) + self.eps)
        kurt = centered.pow(4).mean(dim=1) / (std.pow(4) + self.eps) - 3
        stats = torch.stack((mean, std, skew, kurt), dim=-1)
        if self.num_spectral == 0:
            return stats
        power = torch.fft.rfft(x, dim=1).abs().square()
        power = power[:, 1:, :]
        power = power / (power.sum(dim=1, keepdim=True) + self.eps)
        k = min(self.num_spectral, power.size(1))
        top = power.topk(k, dim=1).values.transpose(1, 2)
        top = F.pad(top, (0, self.num_spectral - k))
        return torch.cat((stats, top), dim=-1)


class UCH(nn.Module):
    def __init__(self, c_max=113, d_model=128, groups=4, d_psi=16):
        super().__init__()
        if c_max < 1 or groups < 1 or d_model % groups:
            raise ValueError("c_max and groups must be positive; groups must divide d_model")
        self.c_max = c_max
        self.descriptor = ChannelDescriptor(d_psi)
        self.group_scorer = nn.Sequential(nn.Linear(d_psi, 32), nn.GELU(), nn.Linear(32, groups))
        self.group_projection = nn.ModuleList(nn.Linear(c_max, d_model // groups) for _ in range(groups))
        self.fusion = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        if x.ndim != 3 or not 1 <= x.size(-1) <= self.c_max:
            raise ValueError(f"Expected [B,T,C] with 1 <= C <= {self.c_max}")
        assignments = self.group_scorer(self.descriptor(x)).softmax(dim=-1)
        padding = self.c_max - x.size(-1)
        x = F.pad(x, (0, padding))
        padded_assignments = F.pad(assignments, (0, 0, 0, padding))
        groups = [projection(x * padded_assignments[:, :, i].unsqueeze(1)) for i, projection in enumerate(self.group_projection)]
        return self.norm(self.fusion(torch.cat(groups, dim=-1))), assignments


class LowRankLinear(nn.Module):
    def __init__(self, input_dim, output_dim, rank=16):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim)

    def forward(self, x):
        return self.up(self.down(x))


class LowRankFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512, rank=16, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(LowRankLinear(d_model, d_ff, rank), nn.GELU(), nn.Dropout(dropout), LowRankLinear(d_ff, d_model, rank), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_size, stride, num_tokens):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, patch_size, stride=stride, groups=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        self.scale_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x, temporal):
        tokens = self.pointwise(self.depthwise(x.transpose(1, 2))).transpose(1, 2)
        return tokens + self.position_embedding + self.scale_embedding + temporal


class AMSPE(nn.Module):
    def __init__(self, d_model, sampling_rate, input_length):
        super().__init__()
        if not math.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive and finite")
        self.input_length = input_length
        self.patch_sizes = dict(zip(SCALES, (max(4, math.floor(.15 * sampling_rate)), max(8, math.floor(.35 * sampling_rate)), max(16, math.floor(.70 * sampling_rate)))))
        if input_length < max(self.patch_sizes.values()):
            raise ValueError("input_length must be at least the largest patch size")
        self.strides = {s: max(p // 2, 2) for s, p in self.patch_sizes.items()}
        self.token_counts = {s: (input_length - p) // self.strides[s] + 1 for s, p in self.patch_sizes.items()}
        self.temporal_projection = nn.Linear(d_model, d_model)
        self.embedding = nn.ModuleDict({s: PatchEmbedding(d_model, self.patch_sizes[s], self.strides[s], self.token_counts[s]) for s in SCALES})

    def forward(self, x):
        if x.size(1) != self.input_length:
            raise ValueError(f"Expected input_length={self.input_length}, received {x.size(1)}")
        temporal = self.temporal_projection(x.mean(dim=1)).unsqueeze(1)
        return {s: self.embedding[s](x, temporal) for s in SCALES}


class CrossAttention(nn.Module):
    def __init__(self, d_model=128, heads=8, attention_dim=16):
        super().__init__()
        if attention_dim % heads:
            raise ValueError("heads must divide attention_dim")
        self.heads = heads
        self.head_dim = attention_dim // heads
        self.Wq = nn.Linear(d_model, attention_dim)
        self.Wk = nn.Linear(d_model, attention_dim)
        self.Wv = nn.Linear(d_model, attention_dim)
        self.out = nn.Linear(attention_dim, d_model)

    def split_heads(self, x):
        return x.reshape(x.size(0), x.size(1), self.heads, self.head_dim).transpose(1, 2)

    def forward(self, query, source):
        source = pool_tokens(source, query.size(1))
        q, k, v = self.split_heads(self.Wq(query)), self.split_heads(self.Wk(source)), self.split_heads(self.Wv(source))
        weights = ((q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(query.size(0), query.size(1), -1))


class SAFCA(nn.Module):
    def __init__(self, token_counts, d_model=128, heads=8, attention_dim=16, rank=16):
        super().__init__()
        self.cross_attention = CrossAttention(d_model, heads, attention_dim)
        self.gates = nn.ModuleDict({s: LowRankLinear(d_model, d_model, rank) for s in SCALES})
        self.norms = nn.ModuleDict({s: nn.LayerNorm(d_model) for s in SCALES})
        self.phi_real = nn.ParameterDict({s: nn.Parameter(torch.ones(1, token_counts[s], d_model)) for s in SCALES})
        self.phi_imag = nn.ParameterDict({s: nn.Parameter(torch.zeros(1, token_counts[s], d_model)) for s in SCALES})

    def forward(self, z):
        mixed = {s: torch.fft.ifft(torch.fft.fft(z[s], dim=1) * torch.complex(self.phi_real[s], self.phi_imag[s]), dim=1).real for s in SCALES}
        output = {}
        for s in SCALES:
            context = torch.stack([self.cross_attention(mixed[s], mixed[t]) for t in SCALES if t != s]).mean(dim=0)
            gate = self.gates[s](z[s]).sigmoid()
            output[s] = self.norms[s](gate * mixed[s] + (1 - gate) * context + z[s])
        return output


class ISIE(nn.Module):
    def __init__(self, d_model=128, d_ff=512, rank=16, dropout=.1):
        super().__init__()
        self.coarse_to_medium = LowRankLinear(d_model, d_model, rank)
        self.medium_to_fine = LowRankLinear(d_model, d_model, rank)
        self.ff_medium = LowRankFFN(d_model, d_ff, rank, dropout)
        self.ff_fine = LowRankFFN(d_model, d_ff, rank, dropout)
        self.norm_medium = nn.LayerNorm(d_model)
        self.norm_fine = nn.LayerNorm(d_model)

    def forward(self, z):
        message = self.coarse_to_medium(pool_tokens(z["c"], z["m"].size(1)))
        medium = self.norm_medium(self.ff_medium(z["m"] + message) + z["m"])
        message = self.medium_to_fine(pool_tokens(medium, z["f"].size(1)))
        fine = self.norm_fine(self.ff_fine(z["f"] + message) + z["f"])
        return {"f": fine, "m": medium, "c": z["c"]}


class TemporalAttention(CrossAttention):
    def __init__(self, d_model=128, heads=8, attention_dim=16, eps=1e-6):
        super().__init__(d_model, heads, attention_dim)
        self.eps = eps

    def forward(self, x):
        q = F.elu(self.split_heads(self.Wq(x))) + 1
        k = F.elu(self.split_heads(self.Wk(x))) + 1
        v = self.split_heads(self.Wv(x))
        kv = k.transpose(-2, -1) @ v
        numerator = q @ kv
        denominator = (q * k.sum(dim=2, keepdim=True)).sum(dim=-1, keepdim=True)
        return self.out((numerator / (denominator + self.eps)).transpose(1, 2).reshape(x.size(0), x.size(1), -1))


class ChannelAttention(nn.Module):
    def __init__(self, num_tokens, heads=8, attention_dim=8, dropout=.1):
        super().__init__()
        self.input_projection = nn.Linear(num_tokens, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, heads, dropout=dropout, batch_first=True)
        self.output_projection = nn.Linear(attention_dim, num_tokens)

    def forward(self, x):
        channels = self.input_projection(x.transpose(1, 2))
        out, _ = self.attention(channels, channels, channels, need_weights=False)
        return self.output_projection(out).transpose(1, 2)


class FTCA(nn.Module):
    def __init__(self, num_tokens, d_model=128, heads=8, d_ff=512, rank=16, dropout=.1):
        super().__init__()
        self.temporal = TemporalAttention(d_model, heads)
        self.channel = ChannelAttention(num_tokens, heads, dropout=dropout)
        self.ffn = LowRankFFN(d_model, d_ff, rank, dropout)
        self.norm_t = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm_t(self.temporal(x) + x)
        x = self.norm_c(self.channel(x) + x)
        return self.norm_ffn(self.ffn(x) + x)


class UniHARBlock(nn.Module):
    def __init__(self, token_counts, d_model=128, heads=8, d_ff=512, rank=16, dropout=.1):
        super().__init__()
        self.safca = SAFCA(token_counts, d_model, heads, rank=rank)
        self.isie = ISIE(d_model, d_ff, rank, dropout)
        self.ftca = nn.ModuleDict({s: FTCA(token_counts[s], d_model, heads, d_ff, rank, dropout) for s in SCALES})

    def forward(self, z):
        z = self.isie(self.safca(z))
        return {s: self.ftca[s](z[s]) for s in SCALES}


class AttentionPooling(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.score = nn.Linear(d_model, 1, bias=False)
        self.normalizer = math.sqrt(d_model)

    def forward(self, x):
        attention = (self.score(x).squeeze(-1) / self.normalizer).softmax(dim=1)
        return (x * attention.unsqueeze(-1)).sum(dim=1), attention


class MultiScaleAggregation(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.pooling = nn.ModuleDict({s: AttentionPooling(d_model) for s in SCALES})
        self.scale_weights = nn.Linear(3 * d_model, 3)

    def forward(self, z):
        values = {s: self.pooling[s](z[s]) for s in SCALES}
        beta = self.scale_weights(torch.cat([values[s][0] for s in SCALES], dim=-1)).softmax(dim=-1)
        fused = (torch.stack([values[s][0] for s in SCALES], dim=1) * beta.unsqueeze(-1)).sum(dim=1)
        return fused, beta, {s: values[s][1] for s in SCALES}


class UniHAR(nn.Module):
    def __init__(self, num_classes, sampling_rate, input_length, c_max=113, d_model=128, groups=4, d_psi=16, depth=4, heads=8, d_ff=512, rank=12, unique_blocks=4, transformer_dropout=.1, classifier_dropout=.2):
        super().__init__()
        if num_classes < 2 or depth < 1 or unique_blocks < 1 or unique_blocks > depth or d_model % heads:
            raise ValueError("Invalid class count, depth, unique_blocks, or head count")
        self.depth = depth
        self.unique_blocks = unique_blocks
        self.uch = UCH(c_max, d_model, groups, d_psi)
        self.amspe = AMSPE(d_model, sampling_rate, input_length)
        self.encoder = nn.ModuleList(UniHARBlock(self.amspe.token_counts, d_model, heads, d_ff, rank, transformer_dropout) for _ in range(unique_blocks))
        self.aggregation = MultiScaleAggregation(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(classifier_dropout), nn.Linear(256, 128), nn.GELU(), nn.Dropout(classifier_dropout), nn.Linear(128, num_classes))

    def forward(self, x, return_aux=False):
        x, assignments = self.uch(x)
        z = self.amspe(x)
        for i in range(self.depth):
            z = self.encoder[i % self.unique_blocks](z)
        representation, beta, attention = self.aggregation(z)
        logits = self.classifier(representation)
        if return_aux:
            return {"logits": logits, "channel_group_assignments": assignments, "scale_weights": beta, "token_attention": attention, "multi_scale_features": z}
        return logits


def scale_utilization_loss(beta):
    return (beta * beta.clamp_min(1e-8).log()).sum(dim=-1).mean()


def unihar_loss(logits, targets, beta, class_weights=None, lambda_su=.1, model=None, lambda_l2=0., label_smoothing=.1):
    loss = F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=label_smoothing)
    loss = loss + lambda_su * scale_utilization_loss(beta)
    if lambda_l2:
        if model is None:
            raise ValueError("model is required when lambda_l2 is nonzero")
        loss = loss + lambda_l2 * sum(p.square().sum() for p in model.parameters() if p.requires_grad)
    return loss
