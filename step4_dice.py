"""Step 4. To draw something new, pick two numbers with dice. The dice follow the standard bell curve:
most rolls are near 0, almost none beyond 3. Which digits come out of the plain network?"""
import torch
from data import XL, XT, YT
from models import get, judge, kinds, rolls

N = 2000


def reach(code):
    """Share of the pictures whose point lies within 3 of the centre of the map, where the dice land."""
    return 100 * ((code ** 2).sum(1).sqrt() <= 3).float().mean().item()


@torch.no_grad()
def run():
    out = {'n': N, 'rolls': rolls(N, 2)[:400].tolist(), 'starts': []}
    for start in range(3):
        net = get(seed=start)
        code = net.squeeze(XL)[0]
        out['starts'].append({'low': code.min(0).values.tolist(), 'high': code.max(0).values.tolist(), 'reach': reach(code),
                              'codes': [[round(v, 3) for v in row] for row in code.tolist()],
                              'kinds': kinds(net.draw(rolls(N, 2)))})
        s = out['starts'][-1]
        print(f"start {start}: map from {min(s['low']):.0f} to {max(s['high']):.0f}, {s['reach']:.0f} % of the pictures within reach of the dice; "
              f"of {N} drawings the judge names {s['kinds']}")
    net = get()
    out['drawn'] = net.draw(rolls(N, 2)[:24]).tolist()
    out['named'] = judge()(net.draw(rolls(N, 2)[:24])).argmax(1).tolist()
    out['judge'] = 100 * (judge()(XT).argmax(1) == YT).float().mean().item()
    print(f"the judge names {out['judge']:.1f} % of the 500 test pictures right")
    out['beyond3'] = 100 * ((rolls(100000, 2) ** 2).sum(1).sqrt() > 3).float().mean().item()
    print(f"share of dice rolls farther than 3 from the centre: {out['beyond3']:.1f} %")
    return out


if __name__ == '__main__':
    run()
