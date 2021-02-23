from dsmnet import dsmdataset

def test_dsmdataset():
    ns = 500
    freq = 0.005
    freq2 = 0.1
    sampling_hz = 1
    tlen = 1638.4
    nspc = 256

    dsmdataset.single_layer_dpp(
        ns, freq, freq2, sampling_hz, tlen, nspc)


if __name__ == '__main__':
    test_dsmdataset()
