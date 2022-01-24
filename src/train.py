from src.data import *
from src.float import *
from src.int import *
from src.param import *

np.random.seed(42)
eta, beta1, beta2 = 0.01, 0.9, 0.999


# arch: number of aggregators per layer, see below:
#   https://github.com/GraphSAINT/GraphSAINT/blob/master/train_config/table2/reddit2_rw.yml#L5
def train(name, epochs, arch, in_size, hidden_size, out_size, scale=0.75, prune=None, multi_class=False, precision=16):
    arch = [a + 1 for a in arch]
    arch.append(1)  # for the classifier
    ws, mws, vws, bs, mbs, vbs = initialize(arch, in_size, hidden_size, out_size, scale, np.float64)
    qws, iws, wfls = quantize_params(ws, precision)
    qbs, ibs, bfls = quantize_params(bs, precision)
    qmws, qvws, qmbs, qvbs = copy_tensor_arrays(mws, vws, mbs, vbs)
    imws, ivws, imbs, ivbs = copy_tensor_arrays(mws, vws, mbs, vbs)

    for epoch in range(epochs):
        x, e, y, nl = load_epoch(name, epoch)
        qx, ix, xfl, qe, ie, efl = quantize_input(x, e, precision)

        fms, norm_denom = forward(arch, x, e, ws, bs, prune)
        qfms, q_norm_denom, agg_fls, fm_fls, nfl = q_forward(arch, qx, qe, qws, qbs, prune, precision)
        ifms, i_norm_denom = i_forward(arch, ix, ie, iws, ibs, xfl, efl, wfls, bfls, agg_fls, fm_fls, nfl, prune,
                                       precision)

        evaluate(epoch, fms[-1][-1], y, multi_class)
        evaluate(epoch, qfms[-1][-1], y, multi_class)
        evaluate(epoch, ifms[-1][-1], y, multi_class)

        grad_ws, grad_bs = backward(arch, e, x, fms, ws, y, nl, norm_denom, prune, multi_class)
        grad_qws, grad_qbs = q_backward(arch, qe, qx, qfms, qws, y, nl, q_norm_denom, prune, multi_class, precision)

        ws, mws, vws, bs, mbs, vbs = adam(arch, ws, mws, vws, bs, mbs, vbs, grad_ws, grad_bs)
        qws, qmws, qvws, qbs, qmbs, qvbs = q_adam(arch, qws, qmws, qvws, qbs, qmbs, qvbs, grad_qws, grad_qbs, precision)


def quantize_params(ws, precision):
    qws, iws, wfls = [[] for _ in ws], [[] for _ in ws], [[] for _ in ws]
    for i, layer_ws in enumerate(ws):
        for w in layer_ws:
            qw, iw, wfl = parallel_search_quantize(w, precision)
            qws[i].append(qw)
            iws[i].append(iw)
            wfls[i].append(wfl)
    return qws, iws, wfls


def load_epoch(name, epoch):
    # sampling done by GraphSAINT and is not our contribution
    x = load(name, epoch, 'x', np.float64)
    e = load(name, epoch, 'e', np.float64, dense=False)
    y = load(name, epoch, 'y', np.float64).astype(np.int32)
    nl = np.array([load(name, epoch, 'norm_loss', np.float64)]).T
    return x, e, y, nl


def quantize_input(x, e, precision):
    ev = e[1]  # e is stored as (indices, values)
    qx, ix, xfl = parallel_search_quantize(x, precision)
    qev, iev, efl = parallel_search_quantize(ev, precision)
    qe, ie = (e[0], qev), (e[0], iev)
    return qx, ix, xfl, qe, ie, efl


# helper function to reduce code duplication
def dense_params(arch, layer, agg, x, e, fms, ws, bs):
    ifm = x if layer == 0 else fms[layer - 1][-1]
    w = ws[layer][agg]
    b = bs[layer][agg]
    layer_e = e if agg == 1 else None
    relu = layer != len(arch) - 1
    return ifm, w, b, layer_e, relu


def forward(arch, x, e, ws, bs, prune):
    fms = []  # intermediate feature maps
    norm_denom = np.zeros(0)

    for layer, layer_size in enumerate(arch):
        layer_fms = []

        for agg in range(layer_size):
            ifm, w, b, layer_e, relu = dense_params(arch, layer, agg, x, e, fms, ws, bs)
            ofm = dense(layer_e, ifm, w, b, relu, prune)
            layer_fms.append(ofm)

        if layer_size == 2:
            layer_fms.append(np.concatenate((layer_fms[0], layer_fms[1]), axis=1))

        if layer == len(arch)-2:
            h, norm_denom = norm(layer_fms[-1])
            layer_fms.append(h)

        fms.append(layer_fms)

    return fms, norm_denom


def q_forward(arch, x, e, ws, bs, prune, precision):
    fms, agg_fls, fm_fls = [], [], []
    norm_denom = np.zeros(0)
    nfl = 0

    for layer, layer_size in enumerate(arch):
        layer_fms, layer_agg_fls, layer_fm_fls = [], [], []

        for agg in range(layer_size):
            ifm, w, b, layer_e, relu = dense_params(arch, layer, agg, x, e, fms, ws, bs)
            ofm, agg_fl, ofm_fl = q_dense(layer_e, ifm, w, b, relu, prune, precision)
            layer_fms.append(ofm)
            layer_agg_fls.append(agg_fl)
            layer_fm_fls.append(ofm_fl)

        if layer_size == 2:
            h = np.concatenate((layer_fms[0], layer_fms[1]), axis=1)
            h, h_fl = search_quantize(h, precision)
            layer_fms.append(h)
            layer_fm_fls.append(h_fl)

        if layer == len(arch)-2:
            h, norm_denom = norm(layer_fms[-1])
            h, nfl = search_quantize(h, precision)
            layer_fms.append(h)

        fms.append(layer_fms)
        agg_fls.append(layer_agg_fls)
        fm_fls.append(layer_fm_fls)

    return fms, norm_denom, agg_fls, fm_fls, nfl


def i_dense_params(arch, layer, agg, xfl, wfls, bfls, agg_fls, fm_fls, nfl):
    ifm_fl = xfl if layer == 0 else fm_fls[layer - 1][-1]
    if layer == len(arch) - 1:
        ifm_fl = nfl
    wfl = wfls[layer][agg]
    bfl = bfls[layer][agg]
    agg_fl = agg_fls[layer][agg]
    ofm_fl = fm_fls[layer][-1]
    return ifm_fl, wfl, bfl, agg_fl, ofm_fl


def i_forward(arch, x, e, ws, bs, xfl, efl, wfls, bfls, agg_fls, fm_fls, nfl, prune, precision):
    fms = []
    norm_denom = np.zeros(0)

    for layer, layer_size in enumerate(arch):
        layer_fms = []

        for agg in range(layer_size):
            ifm, w, b, layer_e, relu = dense_params(arch, layer, agg, x, e, fms, ws, bs)
            ifm_fl, wfl, bfl, agg_fl, ofm_fl = i_dense_params(arch, layer, agg, xfl, wfls, bfls, agg_fls, fm_fls, nfl)
            ofm = i_dense(layer_e, ifm, w, b, efl, ifm_fl, wfl, bfl, agg_fl, ofm_fl, relu, prune, precision)
            layer_fms.append(ofm)

        if layer_size == 2:
            layer_fms.append(np.concatenate((layer_fms[0], layer_fms[1]), axis=1))

        if layer == len(arch)-2:
            full_precision_fm = layer_fms[-1].astype(np.float64) / (2.0 ** fm_fls[layer][-1])
            h, norm_denom = norm(full_precision_fm)
            h = int_quantize(h, nfl, precision)
            layer_fms.append(h)
            fm_fls.append(nfl)

        fms.append(layer_fms)

    return fms, norm_denom


def evaluate(epoch, ofm, y, multi_class):
    correct = 0
    total = 0

    if multi_class:
        result = np.zeros_like(ofm)
        result[ofm > 0] = 1
        for pred_row, truth_row in zip(result, y):
            for pred, truth in zip(pred_row, truth_row):
                correct += (1 if pred == truth else 0)
                total += 1
    else:
        result = np.argmax(ofm, axis=1)
        for pred, truth in zip(result, y):
            correct += (1 if pred == truth else 0)
            total += 1

    print(f"{epoch} {correct}/{total}, {correct / total * 100:.2f}%")


def backward_start(arch, fms, y, nl, multi_class):
    grad_fms = [np.zeros(0) for _ in arch]  # initialize as np.array to silence a pycharm warning
    grad_ws, grad_bs = [[] for _ in arch], [[] for _ in arch]
    grad_fn = binary_categorical_cross_entropy_grad if multi_class else categorical_cross_entropy_grad
    grad_fms[-1] = grad_fn(fms[-1][-1], y, nl)
    return grad_fms, grad_ws, grad_bs


def dense_grad_params(arch, layer, layer_size, agg, e, fms, ws, grad_fms):
    layer_e = e if agg == 1 else None
    w = ws[layer][agg]
    output_size = int(fms[layer][-1].shape[1] / layer_size)
    ofm = fms[layer][-1][:, (agg * output_size):((agg + 1) * output_size)]
    grad_ofm = grad_fms[layer][:, (agg * output_size):((agg + 1) * output_size)]
    relu = layer != len(arch) - 1
    return layer_e, w, ofm, grad_ofm, relu


def backward(arch, e, x, fms, ws, y, nl, norm_denom, prune, multi_class):
    grad_fms, grad_ws, grad_bs = backward_start(arch, fms, y, nl, multi_class)

    for layer in range(len(arch)-1, -1, -1):
        layer_size = arch[layer]
        ifm = fms[layer - 1][-1] if layer > 0 else x
        grad_ifm = np.zeros_like(ifm)

        for agg in range(layer_size):
            layer_e, w, ofm, grad_ofm, relu = dense_grad_params(arch, layer, layer_size, agg, e, fms, ws, grad_fms)
            grad_agg, grad_w, grad_b = dense_grad(layer_e, ifm, w, ofm, grad_ofm, relu, prune)
            grad_ws[layer].append(grad_w)
            grad_bs[layer].append(grad_b)
            grad_ifm += grad_agg

        if layer > 0:
            grad_fms[layer-1] = grad_ifm

        if layer == len(arch)-1:
            grad_fms[layer-1] = norm_grad(fms[layer-1][0], norm_denom, grad_fms[layer-1])

    return grad_ws, grad_bs


def q_backward(arch, e, x, fms, ws, y, nl, norm_denom, prune, multi_class, precision):
    grad_fms, grad_ws, grad_bs = backward_start(arch, fms, y, nl, multi_class)
    grad_fms[-1], _ = search_quantize(grad_fms[-1], precision)

    for layer in range(len(arch)-1, -1, -1):
        layer_size = arch[layer]
        ifm = fms[layer - 1][-1] if layer > 0 else x
        grad_ifm = np.zeros_like(ifm)

        for agg in range(layer_size):
            layer_e, w, ofm, grad_ofm, relu = dense_grad_params(arch, layer, layer_size, agg, e, fms, ws, grad_fms)
            grad_agg, grad_w, grad_b = q_dense_grad(layer_e, ifm, w, ofm, grad_ofm, relu, prune)
            grad_ws[layer].append(grad_w)
            grad_bs[layer].append(grad_b)
            grad_ifm += grad_agg

        if layer > 0:
            grad_fms[layer-1] = grad_ifm

        if layer == len(arch)-1:
            grad_fms[layer-1] = norm_grad(fms[layer-1][0], norm_denom, grad_fms[layer-1])
            grad_fms[layer-1], _ = search_quantize(grad_fms[layer-1], precision)

    return grad_ws, grad_bs


def adam(arch, ws, mws, vws, bs, mbs, vbs, grad_ws, grad_bs):
    for layer, layer_size in enumerate(arch):
        for agg in range(layer_size):
            ws[layer][agg], mws[layer][agg], vws[layer][agg] = \
                adam_update(ws[layer][agg], mws[layer][agg], vws[layer][agg],
                            grad_ws[layer][agg], eta, beta1, beta2)
            bs[layer][agg], mbs[layer][agg], vbs[layer][agg] = \
                adam_update(bs[layer][agg], mbs[layer][agg], vbs[layer][agg],
                            grad_bs[layer][agg], eta, beta1, beta2)
    return ws, mws, vws, bs, mbs, vbs


def q_adam(arch, ws, mws, vws, bs, mbs, vbs, grad_ws, grad_bs, precision):
    for layer, layer_size in enumerate(arch):
        for agg in range(layer_size):
            ws[layer][agg], mws[layer][agg], vws[layer][agg] = \
                q_adam_update(ws[layer][agg], mws[layer][agg], vws[layer][agg],
                              grad_ws[layer][agg], eta, beta1, beta2, precision)
            bs[layer][agg], mbs[layer][agg], vbs[layer][agg] = \
                q_adam_update(bs[layer][agg], mbs[layer][agg], vbs[layer][agg],
                              grad_bs[layer][agg], eta, beta1, beta2, precision * 2)
    return ws, mws, vws, bs, mbs, vbs
