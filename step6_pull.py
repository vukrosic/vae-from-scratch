"""Step 6. Second change: a penalty pulls every cloud toward centre 0 and spread 1. Together with the dice
this is the VAE (Kingma and Welling, 2013). Now pick two numbers with dice and draw."""
import torch
from data import XL, XT
from models import get, kinds, rolls, nearest, pull
from step1_squeeze import ink_off
from step4_dice import reach, N

STRENGTH = 0.1   # how much the penalty counts next to the copy error


@torch.no_grad()
def run():
    net = get(dice=True, strength=STRENGTH)
    centre, spread = net.squeeze(XL)
    drawn = net.draw(rolls(N, 2))
    far, which = nearest(drawn)
    examples = [(0.0, 1.0), (2.0, 1.0), (0.0, 0.1), (2.0, 0.1)]
    out = {
        'strength': STRENGTH,
        'penalty': [[c, s, pull(torch.tensor(c), torch.tensor(s)).item()] for c, s in examples],
        'codes': centre.tolist(), 'spreads': spread.mean(1).tolist(),
        'low': centre.min(0).values.tolist(), 'high': centre.max(0).values.tolist(),
        'width': centre.std(0).tolist(), 'spread': spread.mean().item(), 'reach': reach(centre),
        'kinds': kinds(drawn), 'kinds_starts': [kinds(get(dice=True, strength=STRENGTH, seed=i).draw(rolls(N, 2))) for i in range(3)],
        'off': ink_off(net, XT),
        'more_copy': [[round(v, 3) for v in net.draw(net.squeeze(XT[i:i + 1])[0])[0].tolist()] for i in range(1, 9)],
        'drawn': drawn[:24].tolist(), 'drawn_code': rolls(N, 2)[:24].tolist(), 'closest': XL[which[:24]].tolist(), 'far24': far[:24].tolist(),
        'far': far.mean().item(), 'far_min': far.min().item(), 'real_far': nearest(XT)[0].mean().item(),
        'grid': [[net.draw(torch.tensor([[float(x), float(y)]]))[0].tolist() for x in (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2)] for y in (2, 1.5, 1, .5, 0, -.5, -1, -1.5, -2)],
    }
    for c, s, p in out['penalty']:
        print(f'centre {c}, spread {s}: penalty {p:.2f}')
    print(f"map from {min(out['low']):.1f} to {max(out['high']):.1f}; average spread {out['spread']:.2f}; {out['reach']:.0f} % of the pictures within reach of the dice")
    print(f"of {N} drawings the judge names {out['kinds']}")
    print(f"a drawing is {out['far']:.2f} away from the most similar real picture on average (a new real picture: {out['real_far']:.2f}); "
          f"the closest of all {N} is {out['far_min']:.2f} away, so none is a copy")
    return out


if __name__ == '__main__':
    run()
