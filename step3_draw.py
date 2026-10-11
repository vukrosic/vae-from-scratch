"""Step 3. The drawer accepts any two numbers, also a point where no picture sits. Walk on the map from a 3 to a 2
and draw at every stop. Compare the halfway drawing with the lazy way: lay the two pictures over each other."""
import torch
from data import XL, XT
from models import get, judge, nearest

FROM, TO, STOPS = 0, 14, 9   # test picture 0 is a 3, test picture 14 is a 2


@torch.no_grad()
def walk(net):
    a, b = net.squeeze(XT[FROM:FROM + 1])[0], net.squeeze(XT[TO:TO + 1])[0]
    points = torch.cat([a + (b - a) * t / (STOPS - 1) for t in range(STOPS)])
    return points, net.draw(points)


@torch.no_grad()
def run():
    points, drawn = walk(get())
    far, which = nearest(drawn)
    ghost = (XT[FROM] + XT[TO]) / 2                       # both pictures laid over each other, half ink each
    out = {
        'a': XT[FROM].tolist(), 'b': XT[TO].tolist(), 'points': points.tolist(), 'drawn': drawn.tolist(),
        'named': judge()(drawn).argmax(1).tolist(), 'far': far.tolist(),
        'half_far': far[STOPS // 2].item(), 'ghost': ghost.tolist(), 'ghost_far': nearest(ghost[None])[0].item(),
        'real_far': nearest(XT)[0].mean().item(),
        'third_closest': XL[which[2]].tolist(), 'others': XL[:23].tolist(),   # for the picture of how the distance is found
    }
    print('the judge names the 9 drawings on the way:', out['named'])
    print(f"halfway on the map: {out['half_far']:.2f} away from the most similar real picture")
    print(f"the two pictures laid over each other: {out['ghost_far']:.2f} away. A new real picture: {out['real_far']:.2f} on average")
    return out


if __name__ == '__main__':
    run()
