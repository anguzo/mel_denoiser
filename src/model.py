import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:seq_len, :].unsqueeze(0)


class ConvFeedForward(nn.Module):
    def __init__(self, emb_size: int, hidden_size: int, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            emb_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(
            hidden_size, emb_size, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1(x_conv)
        x_conv = self.activation(x_conv)
        x_conv = self.conv2(x_conv)
        return x_conv.transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size: int,
        hidden_size: int,
        kernel_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(emb_size)
        self.ffn = ConvFeedForward(emb_size, hidden_size, kernel_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class MelEmbedder(nn.Module):
    def __init__(self, c_bin: int, c_emb: int, c_hidden: int, kernel_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(
            c_bin, c_hidden, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(
            c_hidden, c_emb, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.pos_enc = PositionalEncoding(c_emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # -> [B, c_hidden, time]
        x = self.activation(x)
        x = self.conv2(x)  # -> [B, c_emb, time]
        x = x.transpose(1, 2)  # -> [B, time, c_emb]
        x = x + self.pos_enc(x)
        return x


class MelDenoiser(L.LightningModule):
    def __init__(
        self,
        n_layers: int = 12,
        c_bin: int = 128,
        c_emb: int = 384,
        kernel_size: int = 9,
        num_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-5,
        lr_decay: float = 0.999999,
    ):
        super().__init__()
        self.save_hyperparameters()
        c_hidden = c_emb * 4

        self.mel_embedder = MelEmbedder(c_bin, c_emb, c_hidden, kernel_size)

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(c_emb, c_hidden, kernel_size, num_heads, dropout)
                for _ in range(n_layers)
            ]
        )

        self.output_layer = nn.Linear(c_emb, c_bin)

        self.lr = lr
        self.lr_decay = lr_decay

    def forward(self, mel_input: torch.Tensor) -> torch.Tensor:
        x = self.mel_embedder(mel_input)
        for block in self.transformer_blocks:
            x = block(x)
        mel_out = self.output_layer(x)  # [B, time, c_emb] -> [B, time, c_bin]
        mel_out = mel_out.transpose(1, 2)  # -> [B, c_bin, time]
        return mel_out

    def common_step(self, batch, batch_idx):
        wet_mel, dry_mel = batch
        denoised_mel = self(wet_mel)
        loss = F.l1_loss(denoised_mel, dry_mel)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("valid/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("test/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: self.lr_decay**step
        )
        return [optimizer], [
            {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        ]
