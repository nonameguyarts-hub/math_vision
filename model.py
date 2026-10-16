"""model.py - the image -> LaTeX network, copied from Step 13 of the training notebook.

Only change: the special-token ids (PAD, SOS, EOS) are constructor arguments instead of notebook
globals. Layer names and shapes are untouched, so best.pt loads with strict=True.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(True)]

class Encoder(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(*conv_block(1, 32), nn.MaxPool2d(2), *conv_block(32, 64), nn.MaxPool2d(2),
                                 *conv_block(64, 128), nn.MaxPool2d(2), *conv_block(128, d), nn.MaxPool2d(2))
    def pos2d(self, h, w, device):
        d4 = self.d // 4
        div = torch.exp(torch.arange(d4, device=device).float() * (-math.log(10000.0) / d4))
        ys = torch.arange(h, device=device).float()[:, None] * div[None]
        xs = torch.arange(w, device=device).float()[:, None] * div[None]
        pe_y, pe_x = torch.cat([ys.sin(), ys.cos()], 1), torch.cat([xs.sin(), xs.cos()], 1)
        pe = torch.cat([pe_y[:, None, :].expand(h, w, -1), pe_x[None, :, :].expand(h, w, -1)], 2)
        return pe.permute(2, 0, 1)
    def forward(self, x, widths):
        f = self.net(x); B, d, h, w = f.shape
        f = f + self.pos2d(h, w, f.device)[None].to(f.dtype)
        wl = torch.clamp((widths + 15) // 16, max=w)
        colmask = torch.arange(w, device=f.device)[None, :] < wl[:, None]
        mask = colmask[:, None, :].expand(B, h, w).reshape(B, h * w)
        return f.flatten(2).transpose(1, 2), mask

class Decoder(nn.Module):
    def __init__(self, V, pad, sos, eos, d=256, emb=128, hid=256, att=256):
        super().__init__()
        self.pad, self.sos, self.eos = pad, sos, eos
        self.emb = nn.Embedding(V, emb, padding_idx=pad)
        self.init_h = nn.Linear(d, hid)
        self.gru = nn.GRUCell(emb + d, hid)
        self.W_f = nn.Linear(d, att, bias=False); self.W_h = nn.Linear(hid, att)
        self.W_c = nn.Linear(1, att, bias=False); self.v = nn.Linear(att, 1, bias=False)
        self.out = nn.Linear(hid + d + emb, 256); self.drop = nn.Dropout(0.3); self.fc = nn.Linear(256, V)
    def start(self, feats, mask):
        m = mask.float().unsqueeze(-1)
        mean = (feats.float() * m).sum(1) / m.sum(1).clamp(min=1)
        h = torch.tanh(self.init_h(mean)).float()
        cov = torch.zeros(feats.size(0), feats.size(1), device=feats.device)
        ctx = torch.zeros(feats.size(0), feats.size(2), device=feats.device)
        return h, ctx, cov, self.W_f(feats)
    def step(self, y_t, h, ctx, cov, pf, feats, mask):
        e = self.emb(y_t)
        h = self.gru(torch.cat([e, ctx], 1), h).float()
        en = self.v(torch.tanh(pf + self.W_h(h)[:, None, :] + self.W_c(cov[:, :, None]))).squeeze(-1)
        a = torch.softmax(en.float().masked_fill(~mask, -1e9), -1)
        ctx = torch.bmm(a[:, None, :], feats).squeeze(1).float()
        logits = self.fc(self.drop(F.relu(self.out(torch.cat([h, ctx, e.float()], 1)))))
        return logits.float(), h, ctx, cov + a
    def forward(self, feats, mask, y_in):
        h, ctx, cov, pf = self.start(feats, mask); outs = []
        for t in range(y_in.size(1)):
            lg, h, ctx, cov = self.step(y_in[:, t], h, ctx, cov, pf, feats, mask); outs.append(lg)
        return torch.stack(outs, 1)
    @torch.no_grad()
    def greedy(self, feats, mask, max_len=210):
        B = feats.size(0); h, ctx, cov, pf = self.start(feats, mask)
        y = torch.full((B,), self.sos, dtype=torch.long, device=feats.device)
        done = torch.zeros(B, dtype=torch.bool, device=feats.device); out = []
        for _ in range(max_len):
            lg, h, ctx, cov = self.step(y, h, ctx, cov, pf, feats, mask)
            y = lg.argmax(-1); y = torch.where(done, torch.full_like(y, self.pad), y)
            out.append(y); done |= (y == self.eos)
            if done.all(): break
        return torch.stack(out, 1)

class Im2Latex(nn.Module):
    def __init__(self, V, pad, sos, eos):
        super().__init__(); self.enc, self.dec = Encoder(), Decoder(V, pad, sos, eos)
    def forward(self, x, widths, y_in):
        f, m = self.enc(x, widths); return self.dec(f, m, y_in)
    @torch.no_grad()
    def predict(self, x, widths, max_len=210):
        f, m = self.enc(x, widths); return self.dec.greedy(f, m, max_len)
