"""The pictures. 1,797 small handwritten digits (8 by 8 squares, shipped inside scikit-learn).
Each square holds how much ink is in it, from 0 (white) to 1 (black). No download."""
import torch
from sklearn.datasets import load_digits

_d = load_digits()
X = torch.tensor(_d.data / 16.0, dtype=torch.float32)   # 1,797 pictures, each a row of 64 squares
Y = torch.tensor(_d.target)                             # the digit a person wrote, 0 to 9
TRAIN = 1297                                            # pictures to learn from; the other 500 are the test

_order = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
LEARN, TEST = _order[:TRAIN], _order[TRAIN:]
XL, YL = X[LEARN], Y[LEARN]                             # the 1,297 pictures every network learns from
XT, YT = X[TEST], Y[TEST]                               # the 500 pictures no network ever learns from
