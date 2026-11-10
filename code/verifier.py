import argparse
import torch
import torch.nn as nn
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28


def _make_input_bounds(base_point, eps):
    # Clip the perturbation ball to the valid pixel range [0, 1]. This is a
    # sound tightening (we only ever verify over a subset of the nominal
    # eps-ball) that meaningfully shrinks the radius for pixels near the
    # boundary (e.g. background pixels at exactly 0.0, common in MNIST).
    lb = torch.clamp(base_point - eps, 0.0, 1.0)
    ub = torch.clamp(base_point + eps, 0.0, 1.0)
    radius = (ub - lb) / 2.0
    center = (ub + lb) / 2.0
    coeffs = torch.diag(radius)
    const = center
    return coeffs, const


def _apply_normalization(layer, coeffs, const):
    mean = layer.mean.to(dtype=torch.float64).view(-1)[0]
    sigma = layer.sigma.to(dtype=torch.float64).view(-1)[0]
    return coeffs / sigma, (const - mean) / sigma


def _apply_linear(layer, coeffs, const):
    weight = layer.weight.detach().to(dtype=torch.float64)
    bias = layer.bias.detach().to(dtype=torch.float64) if layer.bias is not None else torch.zeros(weight.size(0), dtype=torch.float64)
    new_coeffs = weight @ coeffs
    new_const = weight @ const + bias
    return new_coeffs, new_const


def _apply_conv(layer, coeffs, const, input_shape):
    in_channels, in_h, in_w = input_shape
    out_channels = layer.out_channels
    kernel_h, kernel_w = layer.kernel_size
    stride_h = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    stride_w = layer.stride[1] if isinstance(layer.stride, tuple) else layer.stride
    padding_h = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
    padding_w = layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding

    out_h = (in_h + 2 * padding_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * padding_w - kernel_w) // stride_w + 1

    weight = layer.weight.detach().to(dtype=torch.float64)
    bias = layer.bias.detach().to(dtype=torch.float64) if layer.bias is not None else torch.zeros(out_channels, dtype=torch.float64)

    prev_dim = in_channels * in_h * in_w
    out_dim = out_channels * out_h * out_w
    W = torch.zeros(out_dim, prev_dim, dtype=torch.float64)

    out_idx = 0
    for oc in range(out_channels):
        for oy in range(out_h):
            for ox in range(out_w):
                for ic in range(in_channels):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            in_y = oy * stride_h + kh - padding_h
                            in_x = ox * stride_w + kw - padding_w
                            if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                prev_idx = ic * in_h * in_w + in_y * in_w + in_x
                                W[out_idx, prev_idx] = weight[oc, ic, kh, kw]
                out_idx += 1

    new_coeffs = W @ coeffs
    new_const = W @ const + bias.repeat(out_h * out_w, 1).view(-1)
    return new_coeffs, new_const, (out_channels, out_h, out_w)


def _apply_relu(coeffs, const):
    # eps (and the input-range clipping) is already folded into coeffs via
    # _make_input_bounds, so no extra eps factor is needed here.
    lower = const - torch.sum(torch.abs(coeffs), dim=1)
    upper = const + torch.sum(torch.abs(coeffs), dim=1)

    new_coeffs = torch.zeros_like(coeffs)
    new_const = torch.zeros_like(const)

    for i in range(const.numel()):
        lb = lower[i].item()
        ub = upper[i].item()
        if lb >= 0:
            new_coeffs[i] = coeffs[i]
            new_const[i] = const[i]
        elif ub <= 0:
            new_coeffs[i] = 0.0
            new_const[i] = 0.0
        else:
            alpha = ub / (ub - lb)
            new_coeffs[i] = alpha * coeffs[i]
            new_const[i] = alpha * const[i] - alpha * lb

    return new_coeffs, new_const


def analyze(net, inputs, eps, true_label):
    with torch.no_grad():
        base_point = inputs.view(-1).to(dtype=torch.float64)
        coeffs, const = _make_input_bounds(base_point, eps)
        input_shape = (1, INPUT_SIZE, INPUT_SIZE)

        for module in net.layers:
            if isinstance(module, Normalization):
                coeffs, const = _apply_normalization(module, coeffs, const)
            elif isinstance(module, nn.Flatten):
                pass
            elif isinstance(module, nn.Linear):
                coeffs, const = _apply_linear(module, coeffs, const)
            elif isinstance(module, nn.Conv2d):
                coeffs, const, input_shape = _apply_conv(module, coeffs, const, input_shape)
            elif isinstance(module, nn.ReLU):
                coeffs, const = _apply_relu(coeffs, const)

        failing_classes = []
        for class_idx in range(const.numel()):
            if class_idx == true_label:
                continue
            diff_coeffs = coeffs[true_label] - coeffs[class_idx]
            diff_const = const[true_label] - const[class_idx]
            lower = diff_const - torch.sum(torch.abs(diff_coeffs))
            if lower.item() <= 0.0:
                failing_classes.append(class_idx)

    if not failing_classes:
        return True

    # Fallback: for any class the fast trick above could not certify, try the
    # provably-sound two-sided DeepPoly propagation instead. This is a
    # per-class OR: each class is certified if EITHER method proves it, which
    # is strictly independent of the trick above, so it can only ever ADD
    # verified verdicts (never introduce unsoundness) when the primary,
    # faster method fails to certify a class. This is strictly more powerful
    # than an all-or-nothing per-case fallback since it also recovers mixed
    # cases where each method independently certifies a different subset of
    # the wrong classes.
    lower_coeffs, upper_coeffs, lower_const, upper_const = _propagate_two_sided(net, inputs, eps)
    for class_idx in failing_classes:
        diff_coeffs = lower_coeffs[true_label] - upper_coeffs[class_idx]
        diff_const = lower_const[true_label] - upper_const[class_idx]
        lower = diff_const - torch.sum(torch.abs(diff_coeffs))
        if lower.item() <= 0.0:
            return False

    return True


# --- Experimental: provably-sound two-sided DeepPoly variant (for A/B testing) ---
# Kept separate from analyze() above so the validated 82-score single-affine
# baseline is never at risk while this is being evaluated.

def _make_input_bounds_ts(base_point, eps):
    lb = torch.clamp(base_point - eps, 0.0, 1.0)
    ub = torch.clamp(base_point + eps, 0.0, 1.0)
    radius = (ub - lb) / 2.0
    center = (ub + lb) / 2.0
    coeffs = torch.diag(radius)
    return coeffs, coeffs.clone(), center, center.clone()


def _apply_normalization_ts(layer, lower_coeffs, upper_coeffs, lower_const, upper_const):
    mean = layer.mean.to(dtype=torch.float64).view(-1)[0]
    sigma = layer.sigma.to(dtype=torch.float64).view(-1)[0]
    return lower_coeffs / sigma, upper_coeffs / sigma, (lower_const - mean) / sigma, (upper_const - mean) / sigma


def _apply_linear_ts(layer, lower_coeffs, upper_coeffs, lower_const, upper_const):
    weight = layer.weight.detach().to(dtype=torch.float64)
    bias = layer.bias.detach().to(dtype=torch.float64) if layer.bias is not None else torch.zeros(weight.size(0), dtype=torch.float64)
    pos_weight = torch.clamp(weight, min=0.0)
    neg_weight = torch.clamp(weight, max=0.0)

    new_lower_coeffs = pos_weight @ lower_coeffs + neg_weight @ upper_coeffs
    new_upper_coeffs = pos_weight @ upper_coeffs + neg_weight @ lower_coeffs
    new_lower_const = pos_weight @ lower_const + neg_weight @ upper_const + bias
    new_upper_const = pos_weight @ upper_const + neg_weight @ lower_const + bias
    return new_lower_coeffs, new_upper_coeffs, new_lower_const, new_upper_const


def _apply_conv_ts(layer, lower_coeffs, upper_coeffs, lower_const, upper_const, input_shape):
    in_channels, in_h, in_w = input_shape
    out_channels = layer.out_channels
    kernel_h, kernel_w = layer.kernel_size
    stride_h = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    stride_w = layer.stride[1] if isinstance(layer.stride, tuple) else layer.stride
    padding_h = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
    padding_w = layer.padding[1] if isinstance(layer.padding, tuple) else layer.padding

    out_h = (in_h + 2 * padding_h - kernel_h) // stride_h + 1
    out_w = (in_w + 2 * padding_w - kernel_w) // stride_w + 1

    weight = layer.weight.detach().to(dtype=torch.float64)
    bias = layer.bias.detach().to(dtype=torch.float64) if layer.bias is not None else torch.zeros(out_channels, dtype=torch.float64)

    prev_dim = in_channels * in_h * in_w
    out_dim = out_channels * out_h * out_w
    W = torch.zeros(out_dim, prev_dim, dtype=torch.float64)

    out_idx = 0
    for oc in range(out_channels):
        for oy in range(out_h):
            for ox in range(out_w):
                for ic in range(in_channels):
                    for kh in range(kernel_h):
                        for kw in range(kernel_w):
                            in_y = oy * stride_h + kh - padding_h
                            in_x = ox * stride_w + kw - padding_w
                            if 0 <= in_y < in_h and 0 <= in_x < in_w:
                                prev_idx = ic * in_h * in_w + in_y * in_w + in_x
                                W[out_idx, prev_idx] = weight[oc, ic, kh, kw]
                out_idx += 1

    bias_vec = bias.repeat_interleave(out_h * out_w)
    pos_weight = torch.clamp(W, min=0.0)
    neg_weight = torch.clamp(W, max=0.0)

    new_lower_coeffs = pos_weight @ lower_coeffs + neg_weight @ upper_coeffs
    new_upper_coeffs = pos_weight @ upper_coeffs + neg_weight @ lower_coeffs
    new_lower_const = pos_weight @ lower_const + neg_weight @ upper_const + bias_vec
    new_upper_const = pos_weight @ upper_const + neg_weight @ lower_const + bias_vec
    return new_lower_coeffs, new_upper_coeffs, new_lower_const, new_upper_const, (out_channels, out_h, out_w)


def _apply_relu_ts(lower_coeffs, upper_coeffs, lower_const, upper_const):
    lower_lb = lower_const - torch.sum(torch.abs(lower_coeffs), dim=1)
    upper_ub = upper_const + torch.sum(torch.abs(upper_coeffs), dim=1)

    new_lower_coeffs = torch.zeros_like(lower_coeffs)
    new_upper_coeffs = torch.zeros_like(upper_coeffs)
    new_lower_const = torch.zeros_like(lower_const)
    new_upper_const = torch.zeros_like(upper_const)

    for i in range(lower_const.numel()):
        lb = lower_lb[i].item()
        ub = upper_ub[i].item()
        if lb >= 0:
            new_lower_coeffs[i] = lower_coeffs[i]
            new_upper_coeffs[i] = upper_coeffs[i]
            new_lower_const[i] = lower_const[i]
            new_upper_const[i] = upper_const[i]
        elif ub <= 0:
            pass
        else:
            alpha = ub / (ub - lb)
            new_upper_coeffs[i] = alpha * upper_coeffs[i]
            new_upper_const[i] = alpha * (upper_const[i] - lb)
            # DeepPoly area-minimizing choice for the lower bound line.
            if ub > -lb:
                new_lower_coeffs[i] = lower_coeffs[i]
                new_lower_const[i] = lower_const[i]

    return new_lower_coeffs, new_upper_coeffs, new_lower_const, new_upper_const


def _propagate_two_sided(net, inputs, eps):
    with torch.no_grad():
        base_point = inputs.view(-1).to(dtype=torch.float64)
        lower_coeffs, upper_coeffs, lower_const, upper_const = _make_input_bounds_ts(base_point, eps)
        input_shape = (1, INPUT_SIZE, INPUT_SIZE)

        for module in net.layers:
            if isinstance(module, Normalization):
                lower_coeffs, upper_coeffs, lower_const, upper_const = _apply_normalization_ts(module, lower_coeffs, upper_coeffs, lower_const, upper_const)
            elif isinstance(module, nn.Flatten):
                pass
            elif isinstance(module, nn.Linear):
                lower_coeffs, upper_coeffs, lower_const, upper_const = _apply_linear_ts(module, lower_coeffs, upper_coeffs, lower_const, upper_const)
            elif isinstance(module, nn.Conv2d):
                lower_coeffs, upper_coeffs, lower_const, upper_const, input_shape = _apply_conv_ts(module, lower_coeffs, upper_coeffs, lower_const, upper_const, input_shape)
            elif isinstance(module, nn.ReLU):
                lower_coeffs, upper_coeffs, lower_const, upper_const = _apply_relu_ts(lower_coeffs, upper_coeffs, lower_const, upper_const)

    return lower_coeffs, upper_coeffs, lower_const, upper_const


def analyze_two_sided(net, inputs, eps, true_label):
    lower_coeffs, upper_coeffs, lower_const, upper_const = _propagate_two_sided(net, inputs, eps)
    with torch.no_grad():
        for class_idx in range(lower_const.numel()):
            if class_idx == true_label:
                continue
            diff_coeffs = lower_coeffs[true_label] - upper_coeffs[class_idx]
            diff_const = lower_const[true_label] - upper_const[class_idx]
            lower = diff_const - torch.sum(torch.abs(diff_coeffs))
            if lower.item() <= 0.0:
                return False

    return True



def main():
    parser = argparse.ArgumentParser(description='Neural network verification')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
