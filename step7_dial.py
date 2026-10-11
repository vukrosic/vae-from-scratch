"""Step 7. The strength of the pull is a dial. Turn it from 0 to 2 and watch the copies, the map and the drawings."""
import torch
from data import XL, XT
from models import get, kinds, rolls
from step1_squeeze import ink_off
from step4_dice import N

DIAL = [0.0, 0.1, 0.5, 1.0, 2.0]


@torch.no_grad()
def run():
    out = {'dial': DIAL, 'rows': [], 'average': [round(v, 3) for v in XL.mean(0).tolist()]}
    for strength in DIAL:
        net = get(dice=True, strength=strength)
        centre, spread = net.squeeze(XL)
        drawn = net.draw(rolls(N, 2))
        k = kinds(drawn)
        out['rows'].append({'strength': strength, 'off': ink_off(net, XT), 'width': centre.std(0).tolist(), 'spread': spread.mean(0).tolist(),
                            'used': int((centre.std(0) > 0.1).sum()), 'kinds': k, 'missing': sum(1 for v in k if v < N // 100),
                            'codes': centre[:400].tolist(), 'drawn': drawn[:12].tolist()})
        r = out['rows'][-1]
        print(f"strength {strength}: copies off by {r['off']:.1f}; the map is {r['width'][0]:.2f} and {r['width'][1]:.2f} wide; "
              f"spread {r['spread'][0]:.2f} and {r['spread'][1]:.2f}; numbers still used {r['used']} of 2; digits nearly never drawn: {r['missing']} of 10")
    return out


if __name__ == '__main__':
    run()
