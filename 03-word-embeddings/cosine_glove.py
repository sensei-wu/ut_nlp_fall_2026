from pathlib import Path

import numpy as np

words = {"coffee", "tea", "film", "movie", "democracy"}
vecs = {}
data_path = Path(__file__).parent.parent / "data" / "glove.6B.50d-relativized.txt"
with data_path.open(encoding="utf-8") as f:
    for line in f:
        tok, *vals = line.rstrip().split(" ")
        if tok in words:
            vecs[tok] = np.array(vals, dtype=np.float32)
        if len(vecs) == len(words):
            break

a = vecs["coffee"]
for w in ["tea", "film", "movie", "democracy"]:
    b = vecs[w]
    dot = a @ b
    cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"coffee.{w:<10} dot={dot:7.3f}  cos={cos:6.3f}  |{w}|={np.linalg.norm(b):.2f}")

print("-"*20)

a = vecs["film"]
for w in ["coffee", "tea", "movie", "democracy"]:
    b = vecs[w]
    dot = a @ b
    cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"film.{w:<10} dot={dot:7.3f}  cos={cos:6.3f}  |{w}|={np.linalg.norm(b):.2f}")