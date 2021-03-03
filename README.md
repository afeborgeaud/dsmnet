# dsmnet
1-D seismic imaging (waveform inversion) using convolutional neural networks.<br/>
Inputs are computed using the Python package for seismic waveform computation [dsmpy](https://github.com/afeborgeaud/dsmpy).


<p align="center"><img src="https://github.com/afeborgeaud/dsmnet/blob/main/tests/figures/lenetd3.svg" height="350px"></p>

## Example
The result of a test NNWI for a simple single-layer model of the D" region (lowermost 400 km of the mantle) can be downloaded as a tensorboard log file [here](https://www.dropbox.com/s/k0ir33ltmxaroky/events.out.tfevents.1614158601.merveille.28774.0?dl=1).

To view it, please install tensorboard using conda

```bash
conda install -n your_environment tensorboard
```

From the directory where you downloaded the log file, run
```bash
tensorboard --logdir=.
```
and navigate to http://localhost:6006/.



