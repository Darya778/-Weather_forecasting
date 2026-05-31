# Модель WeatherTransformer
import torch
import torch.nn as nn
import numpy as np


class WeatherTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, max_seq_len=96):
        super().__init__()

        self.region_emb = nn.Embedding(100, 16)
        self.input_size = input_size + 16

        self.input_proj = nn.Linear(self.input_size, d_model)

        self.max_seq_len = max_seq_len
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.classifier = nn.Linear(d_model, 1)
        self.regressor = nn.Linear(d_model, 6)

    def forward(self, x, region_ids=None):
        B, T, F = x.shape

        if T > self.max_seq_len:
            x = x[:, :self.max_seq_len, :]
            T = self.max_seq_len

        if region_ids is not None:
            emb = self.region_emb(region_ids)
            emb = emb.unsqueeze(1).expand(-1, T, -1)
            x = torch.cat([x, emb], dim=2)

        x = self.input_proj(x)
        x = x + self.pos_emb[:, :T, :]

        mask = (x.abs().sum(dim=-1) == 0)
        x = self.transformer(x, src_key_padding_mask=mask)

        logits_seq = self.classifier(x).squeeze(-1)
        reg_seq = self.regressor(x)

        temp = reg_seq[..., 0]
        wind = torch.relu(reg_seq[..., 1])
        humidity = reg_seq[..., 2]
        rain = torch.relu(reg_seq[..., 3])
        pm10 = torch.relu(reg_seq[..., 4])
        pm25 = torch.relu(reg_seq[..., 5])

        reg_seq = torch.stack([temp, wind, humidity, rain, pm10, pm25], dim=-1)

        logits_last = logits_seq[:, -1:]
        reg_last = reg_seq[:, -1:]

        return logits_last, reg_last


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nПАРАМЕТРЫ МОДЕЛИ:")
    print(f"  Всего: {total:,}")
    print(f"  Обучаемых: {trainable:,}")


def estimate_flops(model, input_size, seq_len=32, batch_size=1):
    d_model = model.input_proj.out_features
    nhead = model.transformer.layers[0].self_attn.num_heads
    num_layers = len(model.transformer.layers)
    d_ff = model.transformer.layers[0].linear1.out_features

    flops_proj = (input_size + 16) * d_model * seq_len * 2
    flops_attn = 4 * seq_len * d_model**2 + 2 * seq_len**2 * d_model
    flops_ffn = 2 * seq_len * d_model * d_ff
    flops_per_layer = flops_attn + flops_ffn
    flops_transformer = num_layers * flops_per_layer
    flops_classifier = d_model * 1 * 2
    flops_regressor = d_model * 6 * 2

    total_flops = flops_proj + flops_transformer + flops_classifier + flops_regressor
    total_flops *= batch_size

    if total_flops >= 1e9:
        flops_str = f"{total_flops/1e9:.2f} GFLOPs"
    elif total_flops >= 1e6:
        flops_str = f"{total_flops/1e6:.2f} MFLOPs"
    else:
        flops_str = f"{total_flops/1e3:.2f} KFLOPs"

    print(f"\n FLOPs (при batch_size={batch_size}, seq_len={seq_len}):")
    print(f"   Всего: {flops_str}")
    print(f"   Per layer: {flops_per_layer/1e6:.2f} MFLOPs")
    print(f"   Attention: {flops_attn/1e6:.2f} MFLOPs")
    print(f"   FFN: {flops_ffn/1e6:.2f} MFLOPs")

    return total_flops


def print_model_architecture(model, input_size, seq_len=48):
    print("\nАРХИТЕКТУРА МОДЕЛИ (Transformer):\n")
    print(model)

    try:
        x = torch.randn(1, seq_len, input_size).to(next(model.parameters()).device)
        region = torch.tensor([0]).to(next(model.parameters()).device)

        cls_last, reg_last = model(x, region)

        print("\nПРОВЕРКА ФОРМЫ ВЫХОДОВ:")
        print(f"  Classification last: {cls_last.shape}")
        print(f"  Regression last: {reg_last.shape}")
    except Exception as e:
        print(f"Ошибка при проверке модели: {e}")