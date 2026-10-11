"""Step 5. First change: a picture is no longer one point but a cloud. The squeezer gives a centre and a spread,
and the code is centre + spread x dice roll. The drawer must cope with every point of the cloud. Nothing else changes."""
import torch
from data import XL, XT
from models import get, kinds, rolls
from step1_squeeze import ink_off, SHOWN
from step4_dice import reach, N


@torch.no_grad()
def run():
    net = get(dice=True)
    centre, spread = net.squeeze(XL)
    c, s = net.squeeze(XT[SHOWN:SHOWN + 1])
    out = {
        'spread': spread.mean().item(), 'spreads': [get(dice=True, seed=i).squeeze(XL)[1].mean().item() for i in range(3)],
        'low': centre.min(0).values.tolist(), 'high': centre.max(0).values.tolist(), 'reach': reach(centre),
        'kinds': kinds(net.draw(rolls(N, 2))), 'off': ink_off(net, XT),
        'centre_shown': c[0].tolist(), 'spread_shown': s[0].tolist(),
        'codes': [[round(v, 3) for v in r] for r in centre.tolist()], 'drawn': [[round(v, 3) for v in r] for r in net.draw(rolls(N, 2))[:24].tolist()],
    }
    print(f"after training the average spread is {out['spread']:.5f}: the clouds have shrunk back to points")
    print(f"three starts: {[f'{v:.5f}' for v in out['spreads']]}")
    print(f"map from {min(out['low']):.0f} to {max(out['high']):.0f}, {out['reach']:.0f} % within reach of the dice; drawings: {out['kinds']}")
    return out


if __name__ == '__main__':
    run()
