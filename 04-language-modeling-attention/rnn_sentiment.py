import torch
import torch.nn as nn

torch.manual_seed(0)

# --- Toy data: (sentence, label) 1 = positive, 0 = negative ---
# Note: this example is just for demonstrating RNN framework and architecture; with just 8 examples the model just memorizes the toy dataset and predicts correctly on 
# seen patterns and examples. It will perform very poorly on unseen examples because it cannot generalize. For better example, see rnn_sentiment_glove.py
data = [
    ("the movie was great", 1),
    ("i loved this film", 1),
    ("what a fantastic story", 1),
    ("really good acting", 1),
    ("the movie was terrible", 0),
    ("i hated this film", 0),
    ("what a boring story", 0),
    ("really bad acting", 0),
]

# --- Vocabulary: word -> integer id (0 reserved for padding) ---
vocab = {"<pad>": 0}
for sent, _ in data:
    for w in sent.split():
        vocab.setdefault(w, len(vocab))

def encode(sent, max_len=5):
    ids = [vocab.get(w, 0) for w in sent.split()][:max_len]
    return ids + [0] * (max_len - len(ids))  # pad to fixed length

X = torch.tensor([encode(s) for s, _ in data])     # [batch, seq_len]
y = torch.tensor([label for _, label in data])     # [batch]


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=32, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)  # word -> vector
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)       # the recurrent cell
        self.out = nn.Linear(hidden_dim, num_classes)                  # memory -> scores

    def forward(self, x):
        e = self.embed(x)                     # [batch, seq_len, emb_dim]
        all_h, _ = self.rnn(e)                # memory at every step: [batch, seq_len, hidden]
        lengths = (x != 0).sum(dim=1)         # real (non-pad) length of each sentence
        last = all_h[torch.arange(len(x)), lengths - 1]  # memory at the last REAL word
        return self.out(last)                 # -> [batch, num_classes]


model = RNNClassifier(len(vocab))
loss_fn = nn.CrossEntropyLoss()      # applies softmax internally
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()                  # backprop through time happens here
    opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.3f}")

# --- Try it ---
model.eval()
with torch.no_grad():
    for s in ["the film was great", "really bad story", "i loved this movie", "what a terrible film"]:
        probs = torch.softmax(model(torch.tensor([encode(s)])), dim=-1)[0]
        label = "positive" if probs[1] > 0.5 else "negative"
        print(f"{s!r:25} -> {label} ({probs.max():.2f})")