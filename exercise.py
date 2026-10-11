"""Exercise. Give the VAE one number per picture instead of two: the map becomes a single line.
How many of the ten digits can it still draw from dice? Guess first, then run this file.
Then change K to 3 and to 8 and watch the copy error."""
import torch
from data import XT
from models import get, kinds, rolls
from step1_squeeze import ink_off
from step4_dice import N
from step6_pull import STRENGTH

K = 1


@torch.no_grad()
def run():
    net = get(k=K, dice=True, strength=STRENGTH)
    k = kinds(net.draw(rolls(N, K)))
    line = net.draw(torch.linspace(-2.5, 2.5, 21)[:, None]) if K == 1 else None
    out = {'k': K, 'kinds': k, 'drawn_kinds': sum(1 for v in k if v >= N // 100), 'off': ink_off(net, XT), 'off2': ink_off(get(dice=True, strength=STRENGTH), XT),
           'line': line.tolist() if K == 1 else None}
    print(f"{K} number per picture: of {N} drawings the judge names {k}")
    print(f"digits drawn: {out['drawn_kinds']} of 10. Copies off by {out['off']:.1f} (with 2 numbers: {out['off2']:.1f})")
    return out


if __name__ == '__main__':
    run()
