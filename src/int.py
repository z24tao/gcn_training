import numpy as np


def divide(n, d):
    return np.divide(n, d, out=np.zeros_like(n), where=d != 0)


def aggregate(e, ifm):
    result = np.zeros_like(ifm)
    (ei, ev) = e
    for i in range(ei.shape[1]):
        row, col = ei[0][i], ei[1][i]
        result[row] += ev[i] * ifm[col]
    return result


def relu_grad(ifm, grad_ifm):
    out = np.array(grad_ifm, copy=True)
    out[ifm <= 0] = 0
    return out


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
        aggregation_shift = ifm_fl + wfl - agg_fl
        h = shift(h, aggregation_shift, precision)
        print(f'aggregation shift: {aggregation_shift}')
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


def i_dense_grad(e, ifm, w, ofm, grad_ofm, efl, ifmfl, wfl, gafl, gifmfl, gwfl, gbfl, gofmfl, relu, prune, precision):
    print(efl, ifmfl, wfl, gafl, gifmfl, gwfl, gbfl, gofmfl)
    if relu:
        grad_ofm = relu_grad(ofm, grad_ofm)
    grad_b_shift = gofmfl - gbfl
    print(f'grad bias shift: {grad_b_shift}')
    grad_b = np.sum(grad_ofm, axis=0, dtype=np.int64)
    grad_b = shift(grad_b, grad_b_shift, precision * 2)

    h, hfl = grad_ofm, gofmfl
    if e is not None:
        h = aggregate(e, grad_ofm.astype(np.int64))
        aggregation_shift = efl + gofmfl - gafl
        print(f'aggregation shift: {aggregation_shift}')
        h = shift(h, aggregation_shift, precision)
        hfl = gafl

    grad_w_shift = ifmfl + hfl - gwfl
    print(f'grad weight shift: {grad_w_shift}')
    grad_w = np.matmul(ifm.T, h, dtype=np.int64)
    grad_w = shift(grad_w, grad_w_shift, precision)

    grad_x_shift = hfl + wfl - gifmfl
    print(f'grad ifm shift: {grad_x_shift}')
    grad_x = np.matmul(h, w.T, dtype=np.int64)
    grad_x = shift(grad_x, grad_x_shift, precision)

    return grad_x, grad_w, grad_b


def q_adam_update(x, m, v, g, eta, beta1, beta2, precision):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    x, fl = search_quantize(x - eta * divide(m, np.sqrt(v)), precision)
    return x, m, v, fl


def i_adam_update(x, m, v, g, xfl, gfl, eta, beta1, beta2, precision):
    x = x.astype(np.float64) / (2.0 ** xfl)
    g = g.astype(np.float64) / (2.0 ** gfl)
    x, m, v, fl = q_adam_update(x, m, v, g, eta, beta1, beta2, precision)
    x = int_quantize(x, fl, precision)
    return x, m, v
