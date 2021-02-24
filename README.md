# dsmnet
1-D Neural network waveform inversion (NNWI) using [dsmpy](https://github.com/afeborgeaud/dsmpy)

## Example
The result of a test NNWI for a simple single-layer model of the D" region (lowermost 400 km of the mantle) can be downloaded [here](https://www.dropbox.com/sh/fs84en2063t03vf/AACzWdE1xMw-cQW3NQ5LTc-da?dl=1).

To view it, please install tensorboard,

```bash
conda install -n your_environment tensorboard
```

and run
```bash
tensorboard --logdir=runs
```
from the parent directory of the downloaded 'runs' folder.



