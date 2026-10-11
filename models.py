"""The networks. A squeezer turns 64 squares into a few numbers (the code). A drawer turns a code back into 64 squares.
train() covers the plain autoencoder and the VAE: the only differences are the dice and the pull."""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import XL, YL

NETS = Path(__file__).parent / 'nets'
HIDDEN, STEPS, BATCH, RATE = 128, 6000, 128, 2e-3


def layers(n_in, n_out):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out))


class Net(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k
        self.squeezer = layers(64, 2 * k)   # for each of the k code numbers: a centre and a spread
        self.drawer = layers(k, 64)

    def squeeze(self, x):
        """Picture -> the centre of its code and the spread around that centre."""
        centre, log_spread = self.squeezer(x).chunk(2, -1)
        return centre, log_spread.exp()

    def draw(self, code):
        """Code -> picture. The sigmoid keeps every square between 0 (white) and 1 (black)."""
        return torch.sigmoid(self.drawer(code))


def pull(centre, spread):
    """The penalty of the VAE for one code number. It is 0 only when the centre is 0 and the spread is 1."""
    return 0.5 * (centre ** 2 + spread ** 2 - 1 - 2 * spread.log())


def copy_error(net, x):
    """Squeeze, draw again, and add up the squared differences over the 64 squares. Average over the pictures."""
    return ((net.draw(net.squeeze(x)[0]) - x) ** 2).sum(1).mean()


@torch.enable_grad()
def train(k=2, dice=False, strength=0.0, seed=0):
    """dice=False, strength=0: the plain autoencoder. dice=True: every code is moved by a dice roll times its spread.
    strength > 0: the pull toward centre 0 and spread 1 is added to the loss."""
    torch.manual_seed(seed)
    net = Net(k)
    opt = torch.optim.Adam(net.parameters(), RATE)
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        x = XL[torch.randint(0, len(XL), (BATCH,), generator=g)]
        centre, spread = net.squeeze(x)
        roll = torch.randn(centre.shape, generator=g)            # one dice roll per code number
        code = centre + spread * roll if dice else centre        # the roll is outside the weights: the slope passes through
        miss = ((net.draw(code) - x) ** 2).sum(1).mean()
        loss = miss + strength * pull(centre, spread).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net


def get(k=2, dice=False, strength=0.0, seed=0):
    """A trained network, kept in nets/ so that every step and every run sees the same one."""
    NETS.mkdir(exist_ok=True)
    f = NETS / f'k{k}_dice{int(dice)}_pull{strength}_start{seed}.pt'
    net = Net(k)
    if f.exists():
        net.load_state_dict(torch.load(f))
    else:
        net = train(k, dice, strength, seed)
        torch.save(net.state_dict(), f)
    return net.eval()


@torch.enable_grad()
def judge():
    """A separate network that names the digit in a picture. It is only used to count which digits come out."""
    NETS.mkdir(exist_ok=True)
    f = NETS / 'judge.pt'
    j = layers(64, 10)
    if f.exists():
        j.load_state_dict(torch.load(f))
    else:
        torch.manual_seed(0)
        opt = torch.optim.Adam(j.parameters(), RATE)
        for _ in range(1500):
            i = torch.randint(0, len(XL), (BATCH,))
            loss = F.cross_entropy(j(XL[i]), YL[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.save(j.state_dict(), f)
    return j.eval()


def kinds(pictures):
    """How many of the pictures the judge names 0, 1, ... 9."""
    with torch.no_grad():
        return torch.bincount(judge()(pictures).argmax(1), minlength=10).tolist()


def rolls(n, k, seed=5):
    """n codes picked with dice: every number from the standard bell curve (centre 0, spread 1)."""
    return torch.randn(n, k, generator=torch.Generator().manual_seed(seed))


def nearest(pictures):
    """For each picture: the distance to the most similar picture of the 1,297, and which one it is."""
    d = torch.cdist(pictures, XL)
    return d.min(1).values, d.argmin(1)
