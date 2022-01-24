import numpy as np


def aggregate(e, ifm):
    result = np.zeros_like(ifm)
    (ei, ev) = e
    for i in range(ei.shape[1]):
        row, col = ei[0][i], ei[1][i]
        result[row] += ev[i] * ifm[col]
    return result


def search_quantize(x, wl):
    qxs, diffs = [], []
    for fl in range(-wl, wl * 3):
        qx = quantize(x, fl, wl)
        diff = np.sum(np.abs(x - qx))
        qxs.append(qx)
        diffs.append(diff)

    best = np.argmin(np.array(diffs))
    return qxs[best], best-wl


def quantize(x, fl, wl):
    x = np.round(x * (2.0 ** fl))
    x = np.clip(x, -2 ** (wl - 1), 2 ** (wl - 1) - 1)
    return x / (2.0 ** fl)


def clip(x, wl):
    x = np.clip(x, -2 ** (wl - 1), 2 ** (wl - 1) - 1)
    return x.astype('int' + str(wl))


def int_quantize(x, fl, wl):
    x = np.round(x * (2.0 ** fl))
    return clip(x, wl)


def parallel_search_quantize(x, wl):
    qx, fl = search_quantize(x, wl)
    int_qx = int_quantize(x, fl, wl)
    return qx, int_qx, fl


def shift(x, fl, precision):
    return clip(np.round(x.astype(np.float64) / (2.0 ** fl)), precision)


def i_dense(e, ifm, w, b, e_fl, ifm_fl, wfl, bfl, agg_fl, ofm_fl, relu, prune, precision):
    h = np.matmul(ifm, w, dtype=np.int64)

    if e is not None:
        h = shift(h, ifm_fl + wfl - agg_fl, precision)
        print(f'aggregation shift: {ifm_fl + wfl - agg_fl}')
        h = aggregate(e, h.astype(np.int64))
        bias_shift = bfl - e_fl - agg_fl
        combination_shift = e_fl + agg_fl - ofm_fl
    else:
        bias_shift = bfl - ifm_fl - wfl
        combination_shift = ifm_fl + wfl - ofm_fl

    h = h + shift(b.astype(np.int64), bias_shift, precision * 2)
    print(f'bias shift: {bias_shift}')
    h = shift(h, combination_shift, precision)
    print(f'combination shift: {combination_shift}')

    if relu:
        return np.maximum(h, 0)
    return h
