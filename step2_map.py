"""Step 2. The two numbers of a picture are a point on a flat map. Put all 1,297 pictures on the map.
Nobody told the network which digit a picture shows. Do pictures of the same digit still land together?"""
import torch
from data import XL, YL
from models import get


@torch.no_grad()
def run():
    code = get().squeeze(XL)[0]
    d = torch.cdist(code, code)
    d.fill_diagonal_(1e9)
    same = (YL[d.argmin(1)] == YL).float().mean().item()   # is the closest other point a picture of the same digit?
    near = d.argmin(1)
    mid = code[YL == 3].mean(0)
    i = min((k for k in range(len(YL)) if YL[k] == 3 and YL[near[k]] == 3), key=lambda k: (code[k] - mid).norm().item())
    j = near[i].item()                                      # two neighbours on the map, to look at their pictures
    out = {
        'codes': code.tolist(), 'kinds': YL.tolist(),
        'pair': {'codes': [code[i].tolist(), code[j].tolist()], 'pictures': [XL[i].tolist(), XL[j].tolist()], 'far': d[i, j].item()},
        'low': code.min(0).values.tolist(), 'high': code.max(0).values.tolist(),
        'middle': [code[YL == k].mean(0).tolist() for k in range(10)],
        'same_neighbour': 100 * same,
    }
    print(f"the first number runs from {out['low'][0]:.1f} to {out['high'][0]:.1f}, the second from {out['low'][1]:.1f} to {out['high'][1]:.1f}")
    print(f"for {out['same_neighbour']:.1f} % of the pictures, the closest point on the map is a picture of the same digit")
    return out


if __name__ == '__main__':
    run()
