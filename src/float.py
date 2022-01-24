from src.int import *


def divide(n, d):
    return np.divide(n, d, out=np.zeros_like(n), where=d != 0)


def dense(e, ifm, w, b, relu=True, prune=None):
    if prune is not None:
        w[abs(w) < prune] = 0
    h = ifm @ w
    if e is not None:
        h = aggregate(e, h)
    h = h + b
    if relu:
        return np.maximum(h, 0)
    return h


def q_dense(e, ifm, w, b, relu, prune, precision):
    agg_fl = 0
    if prune is not None:
        w[abs(w) < prune] = 0
    h = ifm @ w
    if e is not None:
        h, agg_fl = search_quantize(h, precision)
        h = aggregate(e, h)
    h = h + b
    h, ofm_fl = search_quantize(h, precision)
    if relu:
        return np.maximum(h, 0), agg_fl, ofm_fl
    return h, agg_fl, ofm_fl


def relu_grad(ifm, grad_ifm):
    out = np.array(grad_ifm, copy=True)
    out[ifm <= 0] = 0
    return out


def dense_grad(e, ifm, w, ofm, grad_ofm, relu=True, prune=None):
    if prune is not None:
        w[abs(w) < prune] = 0
    if relu:
        grad_ofm = relu_grad(ofm, grad_ofm)

    grad_b = np.sum(grad_ofm, axis=0)

    h = grad_ofm
    if e is not None:
        h = aggregate(e, grad_ofm)

    grad_w = ifm.T @ h
    grad_x = h @ w.T
    return grad_x, grad_w, grad_b


def q_dense_grad(e, ifm, w, ofm, grad_ofm, relu=True, prune=None, precision=16):
    if prune is not None:
        w[abs(w) < prune] = 0
    if relu:
        grad_ofm = relu_grad(ofm, grad_ofm)

    grad_b = np.sum(grad_ofm, axis=0)

    h = grad_ofm
    if e is not None:
        h = aggregate(e, grad_ofm)
        h, _ = search_quantize(h, wl=precision)

    grad_w = ifm.T @ h
    grad_x = h @ w.T

    grad_x, _ = search_quantize(grad_x, wl=precision)
    grad_w, _ = search_quantize(grad_w, wl=precision)
    grad_b, _ = search_quantize(grad_b, wl=precision*2)
    return grad_x, grad_w, grad_b


def norm(x):
    p = np.sqrt(np.sum(x * x, axis=1)) + 1e-12
    return x / p[:, None], p


def norm_grad(x, p, g):
    q = np.sum(x * g, axis=1)
    return g / p[:, None] - x * (q / (p ** 3))[:, None]


def softmax(x):
    x -= np.max(x)
    out = np.zeros((len(x), len(x[0])))
    for i, xx in enumerate(x):
        exps = np.exp(xx)
        out[i] = np.zeros_like(exps) if (np.sum(exps) == 0) else (exps / np.sum(exps))
    return out


def categorical_cross_entropy_grad(pred, actual, norm_loss):
    pred = softmax(pred)
    out = np.zeros((len(pred), len(pred[0])), dtype=pred.dtype)
    for i in range(0, len(pred)):
        out[i] = pred[i]
        out[i][actual[i]] -= 1
    return out * norm_loss


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_categorical_cross_entropy_grad(pred, actual, norm_loss):
    return (sigmoid(pred) - actual) * norm_loss


def adam_update(x, m, v, g, eta, beta1, beta2):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    return x - eta * divide(m, np.sqrt(v)), m, v


def q_adam_update(x, m, v, g, eta, beta1, beta2, precision):
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g ** 2
    x, _ = search_quantize(x - eta * divide(m, np.sqrt(v)), precision)
    return x, m, v
