import numpy as np


def initialize_param(shape, scale, dtype):
    shape = [1, shape[0]] if len(shape) == 1 else shape
    p = (np.random.rand(shape[0], shape[1]) * scale * 2 - scale).astype(dtype)
    m = np.zeros(shape).astype(dtype)
    v = np.zeros(shape).astype(dtype)
    return p, m, v


# w: weight, b: bias, m and v are for adam, s means plural
def initialize_layer(order, in_size, out_size, scale, dtype):
    ws, mws, vws, bs, mbs, vbs = [], [], [], [], [], []

    for o in range(order):
        w, mw, vw = initialize_param([in_size, out_size], scale, dtype)
        b, mb, vb = initialize_param([out_size], 0, dtype)
        ws.append(w)
        bs.append(b)
        mws.append(mw)
        mbs.append(mb)
        vws.append(vw)
        vbs.append(vb)

    return ws, mws, vws, bs, mbs, vbs


def initialize(arch, in_size, hidden_size, out_size, scale, dtype):
    ws, mws, vws, bs, mbs, vbs = [], [], [], [], [], []

    for i, order in enumerate(arch):
        layer_in_size = in_size if i == 0 else (hidden_size * arch[i - 1])
        layer_out_size = hidden_size if (i != len(arch) - 1) else out_size
        lws, lmws, lvws, lbs, lmbs, lvbs = initialize_layer(order, layer_in_size, layer_out_size, scale, dtype)
        ws.append(lws)
        bs.append(lbs)
        mws.append(lmws)
        mbs.append(lmbs)
        vws.append(lvws)
        vbs.append(lvbs)

    return ws, mws, vws, bs, mbs, vbs


def copy_tensor_arrays(*tensor_arrays):
    results = []
    for tensor_array in tensor_arrays:
        cp = []
        for tensor in tensor_array:
            cp.append(np.array(tensor, copy=True))
        results.append(cp)
    return results
