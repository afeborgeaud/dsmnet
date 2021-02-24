from dsmpy.utils import modelutils
from dsmpy.dsm import compute_models_parallel
from dsmpy.dataset import Dataset
from dsmpy.utils.cmtcatalog import read_catalog
from dsmpy.event import Event
from dsmpy.station import Station
from dsmpy.windowmaker import WindowMaker
from dsmpy.component import Component
from pytomo.inversion.umcutils import UniformMonteCarlo
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

def _get_events():
    catalog = read_catalog()
    event = Event.event_from_catalog(catalog, '200707211534A')
    event.source_time_function.half_duration = 2.
    return [event]

def _get_stations(event):
    stations = [
        Station(
            '{:03d}'.format(i), 'DSM',
            event.latitude + 70 + 0.5 * i, event.longitude + 0.001)
        for i in range(60)]
    return stations

def _get_dataset_dpp(sampling_hz):
    events = _get_events()
    stations = [_get_stations(event) for event in events]
    dataset = Dataset.dataset_from_arrays(
        events, stations, sampling_hz)
    return dataset

def _get_windows_dpp(dataset):
    phases = ['S', 'Sdiff']
    components = ['T']
    t_before = 14.  # 50
    t_after = 70.   # 30
    windows = WindowMaker.windows_from_dataset(
        dataset, 'ak135', phases, components, t_before, t_after)
    return windows

def _output_to_arr(output, windows, components):
    """

    Returns:
        ndarray: array of shape (1, nsta, nt)

    """
    # TODO sort the stations wrt distance
    nt = int(windows[0].get_length() * output.sampling_hz)
    arr = np.zeros((len(components), len(output.stations), nt))

    for ista, station in enumerate(output.stations):
        windows_filt = [
            window for window in windows
            if (window.station == station
                and window.event == output.event)]
        if len(windows_filt) > 0:
            window_arr = windows_filt[0].to_array()
            i_start = int(window_arr[0] * output.sampling_hz)
            i_end = int(window_arr[1] * output.sampling_hz)
            u_cut = output.us[components, ista, i_start:i_end]
            arr[:, ista, :] = u_cut

    return arr

def _to_X_Y(outputs, windows, freq, freq2, perturbations, model_params):
    """

    Returns:
        ndarray: array of shape (nmod * nev, 1, nsta, nt)
        ndarray: array of shape (nmod * nev, np)

    """

    x_list = []
    y_list = []

    components = [Component.T]

    for imod in range(len(outputs)):
        for iev in range(len(outputs[0])):
            output = outputs[imod][iev]
            output.filter(freq, freq2, 'bandpass')
            output_arr = _output_to_arr(output,
                                        windows, components) # TODO
            x_list.append(output_arr)
            output.free()

            y_arr = _perturbations_to_arr(perturbations[imod], model_params)
            y_list.append(y_arr)

    X = np.array(x_list)
    Y = np.array(y_list)
    return X, Y


def _perturbations_to_arr(perturbation, model_params):
    free_indices = model_params.get_free_indices()
    types = model_params.get_types()
    Y = np.array(perturbation[free_indices])
    return Y

def _plot_models(models):
    fig, ax = plt.subplots(1)
    for model in models:
        model.plot(ax=ax, color='black')
    return fig, ax


def single_layer_dpp(
        ns, freq, freq2, sampling_hz,
        tlen=1638.4, nspc=256, mode=2, seed=0):
    """

    Args:
        ns (int): number of sampled models
        seed (int): seed for the rng (default is 0)

    Returns:

    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        model_ref, model_params, range_dict = modelutils.single_layer_dpp()
        print(model_params.get_mesh_type())
        umcutils = UniformMonteCarlo(
            model_ref, model_params, range_dict,
            mesh_type=model_params.get_mesh_type(), seed=seed)

        models, perturbations = umcutils.sample_models(ns)

        # plot models for quick check
        fig, ax = _plot_models(models)
        ax.set_xlim(6.5, 8.)
        ax.set_ylim(3480., 4000.)
        fig.savefig('./models.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        models = None
        perturbations = None

    dataset = _get_dataset_dpp(sampling_hz)
    outputs = compute_models_parallel(
        dataset, models, tlen, nspc,
        dataset.sampling_hz, mode=mode)

    if rank == 0:
        windows = _get_windows_dpp(dataset)
        X, Y = _to_X_Y(outputs, windows, freq,
                       freq2, perturbations, model_params)

        print(X.shape)
        print(Y.shape)

        with open('X.npy', 'wb') as f:
            np.save(f, X)
        with open('Y.npy', 'wb') as f:
            np.save(f, Y)

        for i in range(X.shape[0]):
            plt.imshow(X[i, 0])
            plt.savefig('X_{}.png'.format(i), bbox_inches='tight')


