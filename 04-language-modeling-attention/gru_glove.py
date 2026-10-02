# GRU text classifier on 20 Newsgroups (2 topics: autos vs space)
# Same architecture as rnn_glove.py, only changed from RNN to GRU

import os
import re
from collections import Counter

import torch
import torch.nn as nn
from sklearn.datasets import fetch_20newsgroups

torch.manual_seed(0)

CATEGORIES = ["rec.autos", "sci.space"]
MAX_LEN = 300        # posts can be 1000+ words, a vanilla RNN won't remember that far anyway
MIN_COUNT = 2        # drop words seen only once (mostly typos / names)
GLOVE_PATH = "data/glove.6B.50d-relativized.txt"   # set to None to train embeddings from scratch
EPOCHS = 8
BATCH = 32

PAD, UNK = 0, 1


# ---- data ----
# remove headers etc, otherwise the model just learns the newsgroup name from the header
train_raw = fetch_20newsgroups(subset="train", categories=CATEGORIES, remove=("headers", "footers", "quotes"))
test_raw = fetch_20newsgroups(subset="test", categories=CATEGORIES, remove=("headers", "footers", "quotes"))


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())[:MAX_LEN]


train = [(tokenize(t), y) for t, y in zip(train_raw.data, train_raw.target)]
test = [(tokenize(t), y) for t, y in zip(test_raw.data, test_raw.target)]
train = [(w, y) for w, y in train if w]   # some posts are empty after stripping
test = [(w, y) for w, y in test if w]
print(f"train {len(train)}  test {len(test)}  labels {train_raw.target_names}")

# vocab from train only, test words we never saw -> UNK
counts = Counter(w for words, _ in train for w in words)
vocab = {"<pad>": PAD, "<unk>": UNK}
for w, c in counts.items():
    if c >= MIN_COUNT:
        vocab[w] = len(vocab)
print("vocab size", len(vocab))


def to_tensor(batch):
    # pad to longest in this batch, not a global length
    longest = max(len(words) for words, _ in batch)
    ids = [[vocab.get(w, UNK) for w in words] + [PAD] * (longest - len(words)) for words, _ in batch]
    labels = [y for _, y in batch]
    return torch.tensor(ids), torch.tensor(labels)


# ---- embeddings ----
EMB_DIM = 50
emb = torch.randn(len(vocab), EMB_DIM) * 0.1
if GLOVE_PATH and os.path.exists(GLOVE_PATH):
    found = 0
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if parts[0] in vocab and len(parts) == EMB_DIM + 1:
                emb[vocab[parts[0]]] = torch.tensor([float(v) for v in parts[1:]])
                found += 1
    print(f"glove: {found}/{len(vocab)} words")
else:
    print("no glove file, embeddings start random")
emb[PAD] = 0


# ---- model ----
class GRUClassifier(nn.Module):
    def __init__(self, emb, hidden=64):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=PAD)
        self.gru = nn.GRU(emb.shape[1], hidden, batch_first=True)  
        self.out = nn.Linear(hidden, 2)

    def forward(self, x):
        h_all, _ = self.gru(self.embed(x))
        lengths = (x != PAD).sum(dim=1).clamp(min=1)
        h_last = h_all[torch.arange(len(x)), lengths - 1]   # memory at the last real word, not at a pad
        return self.out(h_last)


def accuracy(data):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(data), 64):
            x, y = to_tensor(data[i:i + 64])
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(data)


model = GRUClassifier(emb)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ---- train ----
for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(train)).tolist()
    total = 0
    for i in range(0, len(train), BATCH):
        x, y = to_tensor([train[j] for j in order[i:i + BATCH]])
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # rnn gradients can blow up
        opt.step()
        total += loss.item()
    print(f"epoch {epoch + 1}  loss {total / (len(train) / BATCH):.3f}  "
          f"train {accuracy(train):.3f}  test {accuracy(test):.3f}")

# ---- try a few ----
model.eval()
for s in ["the goalie made a great save in the third period", # wrong on purpose to check
          "nasa launched the shuttle into orbit", # should get high score in space
          "the ferrari was slow",  # should get high score in autos
          "there are speed limits" # # should get good score in autos
          ]: 
    x, _ = to_tensor([(tokenize(s), 0)])
    p = torch.softmax(model(x), dim=-1)[0]
    print(f"{s!r:55} -> {CATEGORIES[p.argmax()]} ({p.max():.2f})")