# dsmnet
1-D Neural network waveform inversion (NNWI) using [dsmpy](https://github.com/afeborgeaud/dsmpy)

## Example
The result of a test NNWI for a simple single-layer model of the D" region (lowermost 400 km of the mantle) can be downloaded [here](https://www.dropbox.com/s/k0ir33ltmxaroky/events.out.tfevents.1614158601.merveille.28774.0?dl=1).

To view it, please install tensorboard using conda,

```bash
conda install -n your_environment tensorboard
```

and run
```bash
tensorboard --logdir=runs
```
from the parent directory of the downloaded 'runs' folder.



