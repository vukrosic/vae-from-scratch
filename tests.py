"""Seven checks of the building blocks. Run before the steps: python3 tests.py"""
import torch
from data import X, XL, XT, LEARN, TEST
from models import Net, pull, get, judge, rolls, nearest


def check(name, ok):
    print(('ok   ' if ok else 'FAIL ') + name)
    assert ok, name


with torch.no_grad():
    check('1,797 pictures, 1,297 to learn from and 500 for the test, none in both',
          len(X) == 1797 and len(XL) == 1297 and len(XT) == 500 and not set(LEARN.tolist()) & set(TEST.tolist()))
    check('every square holds ink between 0 and 1', X.min().item() == 0 and X.max().item() == 1)
    net = Net(2)
    centre, spread = net.squeeze(XT[:5])
    check('the squeezer gives 2 centres and 2 spreads per picture, and every spread is above 0', centre.shape == (5, 2) and bool((spread > 0).all()))
    d = net.draw(torch.randn(50, 2) * 30)
    check('the drawer gives 64 squares between 0 and 1 for any two numbers', d.shape == (50, 64) and d.min() >= 0 and d.max() <= 1)
    check('the penalty is 0 at centre 0 and spread 1, and above 0 everywhere else we try',
          pull(torch.tensor(0.), torch.tensor(1.)).item() == 0 and all(pull(torch.tensor(c), torch.tensor(s)) > 0 for c, s in [(1., 1.), (0., .5), (0., 2.), (-3., .1)]))
    p, q = torch.distributions.Normal(torch.tensor(1.3), torch.tensor(0.4)), torch.distributions.Normal(0., 1.)
    check('the penalty is the textbook distance between two bell curves (KL divergence)',
          abs(pull(torch.tensor(1.3), torch.tensor(0.4)) - torch.distributions.kl_divergence(p, q)).item() < 1e-6)
    r = rolls(100000, 2)
    check('the dice follow the standard bell curve: average 0, spread 1', abs(r.mean().item()) < 0.01 and abs(r.std().item() - 1) < 0.01)

c = torch.tensor([0.7], requires_grad=True)
s = torch.tensor([0.2], requires_grad=True)
(c + s * 1.5).backward()   # a code made from a dice roll of 1.5
check('the slope passes through a dice roll: 1 for the centre, the roll itself for the spread', c.grad.item() == 1 and s.grad.item() == 1.5)
