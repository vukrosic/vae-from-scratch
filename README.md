# VAE from scratch

The code for the video "This Network Squeezes a Picture Into Two Numbers. Then It Draws Digits Nobody Wrote".

A picture of a handwritten digit is 64 squares of ink. One small network, the squeezer, turns a picture into two numbers. A second one, the drawer, turns two numbers back into a picture. Both are trained together to copy 1,297 pictures; 500 more are kept hidden for the tests.

Then two dice numbers go into the drawer and it should draw a new digit. With the plain network that fails. Two changes fix it: dice during training, and a penalty that pulls all codes to where the dice fall. That is the variational autoencoder (VAE). The files measure each step.

## Run it

```
pip install torch scikit-learn
./run_all.sh
```

No download and no data set to fetch: the 1,797 pictures are part of scikit-learn. The first run trains about 30 small networks and keeps them in `nets/`; a second run is fast. All numbers below are from a run on a laptop CPU. "Off by" is the average difference in ink per square on a scale from 0 to 100. "Distance" is the distance from a drawing to the most similar real picture; real handwriting from the hidden pictures scores 1.06.

## What each file shows

| file | question | measured |
|---|---|---|
| `step1_squeeze.py` | can 2 numbers hold a picture? | 50,500 weights. Copies of hidden pictures are off by 39.9 before training and 10.6 after |
| `step2_map.py` | where do the codes of all pictures lie? | each digit forms its own island: for 93.4 % of the pictures, the nearest code belongs to the same digit |
| `step3_draw.py` | what does the drawer make from a point nobody used? | halfway between two codes: a digit, distance 0.81. In an empty part of the map: a ghost, distance 1.66 |
| `step4_dice.py` | can dice pick the points? | of 2,000 drawings: 687 sevens, 407 nines, 1 zero, no three. Only 29 % of the picture codes lie where the dice fall |
| `step5_cloud.py` | change 1: add dice to the code during training | the network shrinks the dice away: spread 0.00008. Nothing changes: 400 ones, 940 fives, no two, no three |
| `step6_pull.py` | change 2: a penalty that pulls every code toward the dice | 99 % of the codes lie where the dice fall. All ten digits come out, 160 to 243 times each. The drawings are at distance 0.94, the closest one 0.46, so none is a copy |
| `step7_dial.py` | how strong should the penalty be? | strength 0, 0.1, 0.5, 1, 2: copies off by 10.8, 9.9, 11.3, 16.1, 19.3; digits missing: 4, 0, 0, 4, 9. At 2 every drawing is the same blur |
| `step8_more.py` | 16 code numbers | plain network: copies off by 4.7, dice drawings at distance 1.94. VAE: copies off by 7.0, dice drawings at distance 1.07, all ten digits; it uses 6 of the 16 numbers |
| `exercise.py` | one code number: how many of the ten digits can it still draw? | guess, then run it |
| `tests.py` | eight checks | |

`data.py` loads and splits the pictures. `models.py` holds the squeezer, the drawer, the penalty, the judge and the training loop.

## Limits

Tiny pictures (8 × 8 squares) of ten digits. The judge that names the drawings is our own small network (98 % right on the hidden pictures), and a blur can fool it. The distance to the most similar picture is a rough measure: it does not say whether a digit looks good. A VAE blurs: with 16 code numbers its copies are off by 7.0, the plain network's by 4.7.

## Sources

- Kingma, Welling, "Auto-Encoding Variational Bayes", arXiv 1312.6114 (2013).
- Rezende, Mohamed, Wierstra, "Stochastic Backpropagation and Approximate Inference in Deep Generative Models", arXiv 1401.4082 (2014).
- Burgess, Higgins, Pal, Matthey, Watters, Desjardins, Lerchner, "Understanding disentangling in β-VAE", arXiv 1804.03599 (2018).
- Rombach, Blattmann, Lorenz, Esser, Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models", arXiv 2112.10752 (2021).
