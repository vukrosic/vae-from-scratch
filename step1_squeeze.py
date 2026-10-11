"""Step 1. Squeeze a picture of 64 squares into 2 numbers, then draw it again from those 2 numbers.
The network is trained on one thing only: the drawing should match the picture."""
import torch
from data import XL, XT, YT
from models import Net, get

SHOWN = 0   # the test picture the film follows: a 3


def ink_off(net, x):
    """Average difference per square between picture and copy, on a scale from 0 (white) to 100 (black)."""
    return 100 * (net.draw(net.squeeze(x)[0]) - x).abs().mean().item()


@torch.no_grad()
def run():
    torch.manual_seed(0)
    fresh, net = Net(2), get()
    code = net.squeeze(XT[SHOWN:SHOWN + 1])[0]
    out = {
        'pictures': len(XL) + len(XT), 'learn': len(XL), 'test': len(XT),
        'weights': sum(p.numel() for p in net.parameters()),
        'picture': XT[SHOWN].tolist(), 'kind': YT[SHOWN].item(), 'code': code[0].tolist(), 'copy': net.draw(code)[0].tolist(),
        'fresh_copy': fresh.draw(fresh.squeeze(XT[SHOWN:SHOWN + 1])[0])[0].tolist(),
        'off_fresh': ink_off(fresh, XT), 'off': ink_off(net, XT),
        'square': {'row': 6, 'col': 5, 'ink': XT[SHOWN][6 * 8 + 5].item(), 'copy': net.draw(net.squeeze(XT[SHOWN:SHOWN + 1])[0])[0][6 * 8 + 5].item()},
        'off_shown': ink_off(net, XT[SHOWN:SHOWN + 1]), 'off_shown_fresh': ink_off(fresh, XT[SHOWN:SHOWN + 1]),
        'more': [XT[i].tolist() for i in range(1, 9)], 'more_copy': [net.draw(net.squeeze(XT[i:i + 1])[0])[0].tolist() for i in range(1, 9)],
    }
    print(f"{out['pictures']} pictures: {out['learn']} to learn from, {out['test']} for the test. The network has {out['weights']:,} weights.")
    print(f"the test picture (a {out['kind']}) becomes the two numbers {code[0, 0]:.2f} and {code[0, 1]:.2f}")
    print(f"copy of the 500 test pictures, before training: off by {out['off_fresh']:.1f} of 100 per square. After training: {out['off']:.1f}")
    return out


if __name__ == '__main__':
    run()
