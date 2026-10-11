"""Step 8. Two numbers are few. Give the network 16 numbers per picture. The copies get sharper.
Then pick 16 numbers with dice, once for the plain network and once for the VAE."""
import torch
from data import XL, XT
from models import get, kinds, rolls, nearest
from step1_squeeze import ink_off
from step4_dice import N
from step6_pull import STRENGTH

K = 16


@torch.no_grad()
def run():
    plain, vae = get(k=K), get(k=K, dice=True, strength=STRENGTH)
    roll = rolls(N, K)
    code = plain.squeeze(XL)[0]
    helped = code.mean(0) + code.std(0) * roll        # the dice moved and stretched to where the plain codes sit, number by number
    centre, spread = vae.squeeze(XL)
    d_plain, d_helped, d_vae = plain.draw(roll), plain.draw(helped), vae.draw(roll)
    out = {
        'k': K, 'off2': ink_off(get(), XT), 'off_plain': ink_off(plain, XT), 'off_vae': ink_off(vae, XT),
        'far_plain': nearest(d_plain)[0].mean().item(), 'far_helped': nearest(d_helped)[0].mean().item(),
        'far_vae': nearest(d_vae)[0].mean().item(), 'far_vae_min': nearest(d_vae)[0].min().item(), 'real_far': nearest(XT)[0].mean().item(),
        'kinds_vae': kinds(d_vae), 'used': int((centre.std(0) > 0.1).sum()),
        'width': centre.std(0).tolist(), 'spread': spread.mean(0).tolist(),
        'used_starts': [int((get(k=K, dice=True, strength=STRENGTH, seed=i).squeeze(XL)[0].std(0) > 0.1).sum()) for i in range(3)],
        'drawn_plain': d_plain[:16].tolist(), 'drawn_helped': d_helped[:16].tolist(), 'drawn_vae': d_vae[:16].tolist(),
        'copies': [[XT[i].tolist(), get().draw(get().squeeze(XT[i:i + 1])[0])[0].tolist(), plain.draw(plain.squeeze(XT[i:i + 1])[0])[0].tolist(),
                    vae.draw(vae.squeeze(XT[i:i + 1])[0])[0].tolist()] for i in range(6)],
    }
    print(f"copies off by: {out['off2']:.1f} with 2 numbers, {out['off_plain']:.1f} with 16 numbers (plain), {out['off_vae']:.1f} with 16 numbers (VAE)")
    print(f"drawings from dice, distance to the most similar real picture: plain {out['far_plain']:.2f}, plain with helped dice {out['far_helped']:.2f}, "
          f"VAE {out['far_vae']:.2f}; a new real picture {out['real_far']:.2f}")
    print(f"the VAE draws {out['kinds_vae']}; it uses {out['used']} of its 16 numbers (three starts: {out['used_starts']})")
    return out


if __name__ == '__main__':
    run()
