"""
model.py

Image-to-LaTeX model, image-captioning style:

  encoder:  equation image (1, H, W)  ->  feature grid (C, H', W')  ->  N tokens
  decoder:  attends over the N encoder tokens, produces LaTeX one token at a time.

Encoder: ImageNet-pretrained ResNet-18 truncated at layer3. A from-scratch
CNN (see git history) memorized the training writers and scored ~1% on
unseen writers; pretrained low-level features transfer much better.

Decoder: `TransformerDecoder` (default) or the older `AttentionDecoder`
(GRU + Bahdanau). Both expose forward() / greedy_decode() / beam_search()
so Image2LatexModel is agnostic to the choice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# ImageNet normalization the pretrained backbone expects (grayscale is
# replicated to 3 channels before applying this).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetEncoder(nn.Module):
    """
    (B, 1, H, W) grayscale -> (B, out_channels, H/16, W/16) feature grid,
    using ResNet-18 pretrained on ImageNet, truncated after layer3.

    conv1 + bn1 + layer1 are frozen (generic edge/texture filters);
    layer2 + layer3 fine-tune to handwriting.
    """

    def __init__(self, out_channels: int = 256, pretrained: bool = True,
                 freeze_stem: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1   # 64 ch,  H/4
        self.layer2 = net.layer2   # 128 ch, H/8
        self.layer3 = net.layer3   # 256 ch, H/16
        # Project + normalize: the pretrained backbone's raw activations are
        # low-variance on near-binary handwriting (std ~0.15), which starved
        # the attention mechanism and stalled training. BatchNorm here
        # restores unit-scale features the decoder can actually attend to.
        self.project = nn.Sequential(
            nn.Conv2d(256, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        # Freeze only conv1+bn1 (generic Gabor-like filters); let the rest of
        # the backbone — including its BatchNorm running stats — adapt from
        # ImageNet photos to handwriting.
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False
            self._frozen = [self.stem]
        else:
            self._frozen = []

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen BatchNorm layers in eval mode so their running stats
        # (from ImageNet) don't get overwritten by handwriting batches.
        for m in self._frozen:
            m.eval()
        return self

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.project(x)  # (B, out_channels, H/16, W/16)


# Backwards-compatible alias.
CNNEncoder = ResNetEncoder


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, attn_dim: int = 256):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, attn_dim)
        self.dec_proj = nn.Linear(decoder_dim, attn_dim)
        self.energy = nn.Linear(attn_dim, 1)

    def forward(self, encoder_feats, decoder_hidden):
        """
        encoder_feats: (B, N, encoder_dim)  where N = H' * W'
        decoder_hidden: (B, decoder_dim)
        returns: context (B, encoder_dim), attn_weights (B, N)
        """
        enc = self.enc_proj(encoder_feats)              # (B, N, attn_dim)
        dec = self.dec_proj(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)
        scores = self.energy(torch.tanh(enc + dec)).squeeze(-1)  # (B, N)
        weights = F.softmax(scores, dim=-1)              # (B, N)
        context = torch.bmm(weights.unsqueeze(1), encoder_feats).squeeze(1)  # (B, encoder_dim)
        return context, weights


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, encoder_dim: int = 256, embed_dim: int = 128,
                 hidden_dim: int = 256, pad_id: int = 0, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(embed_dim + encoder_dim, hidden_dim)
        self.init_hidden = nn.Linear(encoder_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def init_state(self, encoder_feats):
        # Mean-pool encoder features to seed the decoder's initial hidden state.
        mean_feat = encoder_feats.mean(dim=1)  # (B, encoder_dim)
        return torch.tanh(self.init_hidden(mean_feat))  # (B, hidden_dim)

    def step(self, prev_token, hidden, encoder_feats):
        """One decode step. prev_token: (B,) long. hidden: (B, hidden_dim)."""
        embedded = self.embedding(prev_token)  # (B, embed_dim)
        context, attn_weights = self.attention(encoder_feats, hidden)  # (B, encoder_dim)
        gru_input = torch.cat([embedded, context], dim=-1)
        hidden = self.gru_cell(gru_input, hidden)
        out = self.dropout(torch.cat([hidden, context, embedded], dim=-1))
        logits = self.output_proj(out)
        return logits, hidden, attn_weights

    def forward(self, encoder_feats, target_seq, teacher_forcing_ratio: float = 1.0):
        """
        Training-time forward pass with teacher forcing.
        target_seq: (B, T) includes <sos> at position 0 and <eos> at the end.
        Returns logits: (B, T-1, vocab_size) predicting tokens 1..T-1.
        """
        batch_size, T = target_seq.shape
        hidden = self.init_state(encoder_feats)
        vocab_size = self.output_proj.out_features
        outputs = torch.zeros(batch_size, T - 1, vocab_size, device=target_seq.device)

        prev_token = target_seq[:, 0]  # <sos>
        for t in range(1, T):
            logits, hidden, _ = self.step(prev_token, hidden, encoder_feats)
            outputs[:, t - 1] = logits
            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            prev_token = target_seq[:, t] if use_teacher else logits.argmax(dim=-1)
        return outputs

    @torch.no_grad()
    def greedy_decode(self, encoder_feats, sos_id: int, eos_id: int,
                      max_len: int = 150, return_attention: bool = False):
        batch_size = encoder_feats.size(0)
        device = encoder_feats.device
        hidden = self.init_state(encoder_feats)
        prev_token = torch.full((batch_size,), sos_id, dtype=torch.long, device=device)

        sequences = [[] for _ in range(batch_size)]
        attentions = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits, hidden, attn = self.step(prev_token, hidden, encoder_feats)
            next_token = logits.argmax(dim=-1)
            for i in range(batch_size):
                if not finished[i]:
                    sequences[i].append(next_token[i].item())
                    if return_attention:
                        attentions[i].append(attn[i].detach().cpu())
                    if next_token[i].item() == eos_id:
                        finished[i] = True
            if finished.all():
                break
            prev_token = next_token
        if return_attention:
            return sequences, attentions
        return sequences

    @torch.no_grad()
    def beam_search(self, encoder_feats, sos_id: int, eos_id: int,
                    beam_size: int = 5, max_len: int = 150, alpha: float = 0.7):
        """
        Length-normalized beam search for a single example
        (encoder_feats: (1, N, C)). Returns a list of token ids.

        alpha is the Google-NMT length penalty exponent; score is
        (sum log p) / (((5 + len) / 6) ** alpha).
        """
        assert encoder_feats.size(0) == 1, "beam_search decodes one example at a time"
        device = encoder_feats.device
        feats = encoder_feats.expand(beam_size, -1, -1)  # (beam, N, C)
        hidden = self.init_state(feats)                    # (beam, hidden)
        tokens = torch.full((beam_size, 1), sos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0
        finished = []  # (score, token_list)

        for _ in range(max_len):
            prev = tokens[:, -1]
            logits, new_hidden, _ = self.step(prev, hidden, feats)
            log_probs = F.log_softmax(logits, dim=-1)          # (beam, V)
            cand = scores.unsqueeze(1) + log_probs             # (beam, V)

            flat = cand.view(-1)
            top_scores, top_idx = flat.topk(beam_size)
            beam_idx = torch.div(top_idx, log_probs.size(-1), rounding_mode="floor")
            tok_idx = top_idx % log_probs.size(-1)

            tokens = torch.cat([tokens[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            hidden = new_hidden[beam_idx]
            scores = top_scores

            keep = []
            for b in range(beam_size):
                if tok_idx[b].item() == eos_id:
                    seq = tokens[b, 1:].tolist()
                    lp = scores[b].item() / (((5 + len(seq)) / 6) ** alpha)
                    finished.append((lp, seq))
                    scores[b] = float("-inf")
                else:
                    keep.append(b)
            if not keep:
                break

        if not finished:
            seq = tokens[scores.argmax(), 1:].tolist()
            finished.append((0.0, seq))
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1]


class _TransformerDecoderLayer(nn.Module):
    """Pre-norm decoder layer that also returns its cross-attention weights."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask, tgt_kpm):
        h = self.norm1(x)
        x = x + self.drop(self.self_attn(h, h, h, attn_mask=tgt_mask,
                                         key_padding_mask=tgt_kpm, need_weights=False)[0])
        h = self.norm2(x)
        att, weights = self.cross_attn(h, mem, mem, need_weights=True,
                                       average_attn_weights=True)
        x = x + self.drop(att)
        h = self.norm3(x)
        x = x + self.drop(self.ff(h))
        return x, weights


class TransformerDecoder(nn.Module):
    """
    Transformer decoder over the flattened encoder feature grid (memory).
    Same interface as AttentionDecoder so Image2LatexModel can use either:
    forward() for training, greedy_decode() / beam_search() for inference.
    """

    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, ff_dim: int = 1024, dropout: float = 0.2,
                 pad_id: int = 0, max_len: int = 200, max_mem: int = 512):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.mem_pos = nn.Parameter(torch.randn(1, max_mem, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            _TransformerDecoderLayer(d_model, nhead, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[pad_id].zero_()

    def _embed(self, tgt):
        T = tgt.size(1)
        return self.drop(self.token_emb(tgt) + self.pos_emb[:, :T])

    def _memory(self, encoder_feats):
        return encoder_feats + self.mem_pos[:, : encoder_feats.size(1)]

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, encoder_feats, target_seq, teacher_forcing_ratio: float = 1.0):
        """target_seq: (B, T) with <sos>..<eos>. Returns logits (B, T-1, vocab)."""
        tgt_in = target_seq[:, :-1]
        mem = self._memory(encoder_feats)
        x = self._embed(tgt_in)
        mask = self._causal_mask(tgt_in.size(1), x.device)
        kpm = tgt_in == self.pad_id
        for layer in self.layers:
            x, _ = layer(x, mem, mask, kpm)
        return self.out(self.norm(x))

    def _step_logits(self, ys, mem):
        """Returns (last-position logits (B, V), last-layer cross-attn (B, T, N))."""
        x = self._embed(ys)
        mask = self._causal_mask(ys.size(1), x.device)
        w_last = None
        for layer in self.layers:
            x, w = layer(x, mem, mask, None)
            w_last = w
        x = self.norm(x)
        return self.out(x[:, -1]), w_last

    @torch.no_grad()
    def greedy_decode(self, encoder_feats, sos_id: int, eos_id: int,
                      max_len: int = 150, return_attention: bool = False):
        B = encoder_feats.size(0)
        device = encoder_feats.device
        mem = self._memory(encoder_feats)
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        seqs = [[] for _ in range(B)]
        attns = [[] for _ in range(B)]
        for _ in range(max_len):
            logits, w = self._step_logits(ys, mem)
            nxt = logits.argmax(dim=-1)
            for i in range(B):
                if not done[i]:
                    seqs[i].append(nxt[i].item())
                    if return_attention:
                        attns[i].append(w[i, -1].detach().cpu())
                    if nxt[i].item() == eos_id:
                        done[i] = True
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            if done.all():
                break
        return (seqs, attns) if return_attention else seqs

    @torch.no_grad()
    def beam_search(self, encoder_feats, sos_id: int, eos_id: int,
                    beam_size: int = 5, max_len: int = 150, alpha: float = 0.7):
        assert encoder_feats.size(0) == 1
        device = encoder_feats.device
        mem = self._memory(encoder_feats).expand(beam_size, -1, -1)
        ys = torch.full((beam_size, 1), sos_id, dtype=torch.long, device=device)
        scores = torch.full((beam_size,), float("-inf"), device=device)
        scores[0] = 0.0
        finished = []
        for _ in range(max_len):
            logp = F.log_softmax(self._step_logits(ys, mem)[0], dim=-1)
            V = logp.size(-1)
            cand = (scores.unsqueeze(1) + logp).view(-1)
            top_s, top_i = cand.topk(beam_size)
            beam_idx = torch.div(top_i, V, rounding_mode="floor")
            tok_idx = top_i % V
            ys = torch.cat([ys[beam_idx], tok_idx.unsqueeze(1)], dim=1)
            scores = top_s
            keep = []
            for b in range(beam_size):
                if tok_idx[b].item() == eos_id:
                    seq = ys[b, 1:].tolist()
                    finished.append((scores[b].item() / (((5 + len(seq)) / 6) ** alpha), seq))
                    scores[b] = float("-inf")
                else:
                    keep.append(b)
            if not keep:
                break
        if not finished:
            finished.append((0.0, ys[scores.argmax(), 1:].tolist()))
        finished.sort(key=lambda z: z[0], reverse=True)
        return finished[0][1]


class Image2LatexModel(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, encoder_channels: int = 256,
                 embed_dim: int = 128, hidden_dim: int = 256,
                 pretrained_encoder: bool = True, decoder: str = "transformer",
                 dec_layers: int = 4, dec_heads: int = 8, dec_ff: int = 1024,
                 dec_dropout: float = 0.2):
        super().__init__()
        self.decoder_type = decoder
        self.encoder = ResNetEncoder(out_channels=encoder_channels,
                                     pretrained=pretrained_encoder)
        if decoder == "transformer":
            self.decoder = TransformerDecoder(
                vocab_size=vocab_size, d_model=encoder_channels, nhead=dec_heads,
                num_layers=dec_layers, ff_dim=dec_ff, dropout=dec_dropout, pad_id=pad_id,
            )
        elif decoder == "gru":
            self.decoder = AttentionDecoder(
                vocab_size=vocab_size, encoder_dim=encoder_channels,
                embed_dim=embed_dim, hidden_dim=hidden_dim, pad_id=pad_id,
            )
        else:
            raise ValueError(f"unknown decoder: {decoder}")

    def encode(self, images):
        feats = self.encoder(images)              # (B, C, H', W')
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, N, C)
        return feats

    def forward(self, images, target_seq, teacher_forcing_ratio: float = 1.0):
        feats = self.encode(images)
        return self.decoder(feats, target_seq, teacher_forcing_ratio)

    @torch.no_grad()
    def predict(self, images, sos_id: int, eos_id: int, max_len: int = 150,
                beam_size: int = 1):
        """
        Decode a batch of images to token-id sequences.
        beam_size == 1 -> batched greedy; beam_size > 1 -> beam search,
        one image at a time (slower, usually more accurate).
        """
        feats = self.encode(images)
        if beam_size <= 1:
            return self.decoder.greedy_decode(feats, sos_id, eos_id, max_len)
        return [
            self.decoder.beam_search(feats[i:i + 1], sos_id, eos_id,
                                     beam_size=beam_size, max_len=max_len)
            for i in range(feats.size(0))
        ]

    @torch.no_grad()
    def predict_with_attention(self, images, sos_id: int, eos_id: int,
                               max_len: int = 150):
        """Greedy decode that also returns per-token attention maps."""
        feats = self.encode(images)
        seqs, attns = self.decoder.greedy_decode(
            feats, sos_id, eos_id, max_len, return_attention=True)
        return seqs, attns

    def encode_grid_shape(self, image_w: int, image_h: int):
        """(H', W') of the encoder feature grid (ResNet-18 through layer3 -> input / 16)."""
        return image_h // 16, image_w // 16
