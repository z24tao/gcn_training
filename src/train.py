from src.data import *
from src.float import *
from src.param import *

np.random.seed(42)
eta, beta1, beta2 = 0.01, 0.9, 0.999


# arch: number of aggregators per layer, see below:
#   https://github.com/GraphSAINT/GraphSAINT/blob/master/train_config/table2/reddit2_rw.yml#L5
def train(name, epochs, arch, in_size, hidden_size, out_size, scale=0.75, prune=None, multi_class=False):
    arch = [a + 1 for a in arch]
    arch.append(1)  # for the classifier
    ws, mws, vws, bs, mbs, vbs = initialize(arch, in_size, hidden_size, out_size, scale, np.float64)

    for epoch in range(epochs):
        x, e, y, nl = load_epoch(name, epoch)
        fms = forward(arch, x, e, ws, bs, prune)
        evaluate(epoch, fms[-1][-1], y, multi_class)
        grad_fms, grad_ws, grad_bs = backward(arch, e, x, fms, ws, y, nl, prune, multi_class)
        ws, mws, vws, bs, mbs, vbs = adam(arch, ws, mws, vws, bs, mbs, vbs, grad_ws, grad_bs)


def load_epoch(name, epoch):
    # sampling done by GraphSAINT and is not our contribution
    x = load(name, epoch, 'x')
    e = load(name, epoch, 'e', dense=False)
    y = load(name, epoch, 'y').astype(np.int32)
    nl = np.array([load(name, epoch, 'norm_loss')]).T
    return x, e, y, nl


def forward(arch, x, e, ws, bs, prune):
    fms = []  # intermediate feature maps

    for layer, layer_size in enumerate(arch):
        layer_fms = []

        for agg in range(layer_size):
            ifm = x if layer == 0 else fms[layer-1][-1]
            w = ws[layer][agg]
            b = bs[layer][agg]
            ofm = dense(e if agg == 1 else None, ifm, w, b, relu=layer != len(arch)-1, prune=prune)
            layer_fms.append(ofm)

        if layer_size == 2:
            layer_fms.append(np.concatenate((layer_fms[0], layer_fms[1]), axis=1))

        if layer == len(arch)-2:
            layer_fms.append(norm(layer_fms[-1]))

        fms.append(layer_fms)

    return fms


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


def backward(arch, e, x, fms, ws, y, nl, prune, multi_class):
    grad_fms = [np.zeros(0) for _ in arch]  # initialize as np.array to silence a pycharm warning
    grad_ws, grad_bs = [[] for _ in arch], [[] for _ in arch]
    grad_fn = binary_categorical_cross_entropy_grad if multi_class else categorical_cross_entropy_grad
    grad_fms[-1] = grad_fn(fms[-1][-1], y, nl)

    for layer in range(len(arch)-1, -1, -1):
        layer_size = arch[layer]
        ifm = fms[layer - 1][-1] if layer > 0 else x
        grad_ifm = np.zeros_like(ifm)

        for agg in range(layer_size):
            layer_e = e if agg == 1 else None
            w = ws[layer][agg]
            output_size = int(fms[layer][-1].shape[1] / layer_size)
            ofm = fms[layer][-1][:, (agg * output_size):((agg + 1) * output_size)]
            grad_ofm = grad_fms[layer][:, (agg * output_size):((agg + 1) * output_size)]
            grad_agg, grad_w, grad_b = dense_grad(layer_e, ifm, w, ofm, grad_ofm, layer != len(arch)-1, prune=prune)
            grad_ws[layer].append(grad_w)
            grad_bs[layer].append(grad_b)
            grad_ifm += grad_agg

        if layer > 0:
            grad_fms[layer-1] = grad_ifm

        if layer == len(arch)-1:
            grad_fms[layer-1] = norm_grad(fms[layer-1][0], grad_fms[layer-1])

    return grad_fms, grad_ws, grad_bs


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
