import math
import cv2
import torch
import torchray
from torchray.attribution.grad_cam import grad_cam
import torchray.benchmark
from torchray.attribution.common import get_module, Probe
from collections import OrderedDict
from PIL import Image
from matplotlib import pyplot as plt

import numpy as np

# A very small number for comparing floating point values.
EPSILON = 1E-9
device = torch.device('cuda:0')
model_arch = "resnet50" #@param ['vgg16', 'resnet50']
model=torchray.benchmark.models.get_model(arch=model_arch, dataset="voc", convert_to_fully_convolutional=False)
# model = torchvision.models.resnet50(pretrained = True)
model=model.to(device)
# print(model)
def gradient_to_grad_cam_saliency(x, baseline=None):
    r"""Convert activation and gradient to a Grad-CAM saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s`: given by:

    .. math::

        s_{n1u} = \max\{0, \sum_{c}x_{ncu}\cdot dx_{ncu}\}

    Args:
        x (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: saliency map.
    """
    if baseline is None:
        baseline = x*0.
    # print(x.grad)
    # Apply global average pooling (GAP) to gradient.
    grad_weight = torch.mean(x.grad, (2, 3), keepdim=True)
    # Linearly combine activations and GAP gradient weights.
    saliency_map = torch.sum(x * grad_weight, 1, keepdim=True)
    # print(saliency_map)
    return saliency_map


class NullContext(object):

    def __init__(self):
        r"""Null context.

        This context does nothing.
        """

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return False


def get_backward_gradient(pred_y, y):
    r"""
    Returns a gradient tensor that is either equal to :attr:`y` (if y is a
    tensor with the same shape as pred_y) or a one-hot encoding in the channels
    dimension.

    :attr:`y` can be either an ``int``, an array-like list of integers,
    or a tensor. If :attr:`y` is a tensor with the same shape as
    :attr:`pred_y`, the function returns :attr:`y` unchanged.

    Otherwise, :attr:`y` is interpreted as a list of class indices. These
    are first unfolded/expanded to one index per batch element in
    :attr:`pred_y` (i.e. along the first dimension). Then, this list
    is further expanded to all spatial dimensions of :attr:`pred_y`.
    (i.e. all but the first two dimensions of :attr:`pred_y`).
    Finally, the function return a "gradient" tensor that is a one-hot
    indicator tensor for these classes.

    Args:
        pred_y (:class:`torch.Tensor`): model output tensor.
        y (int, :class:`torch.Tensor`, list, or :class:`np.ndarray`): target
            label(s) that can be cast to :class:`torch.long`.

    Returns:
        :class:`torch.Tensor`: gradient tensor with the same shape as
            :attr:`pred_y`.
    """
    # print(len(pred_y.shape))
    assert isinstance(pred_y, torch.Tensor)
    # print(y)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long, device=pred_y.device)
    assert isinstance(y, torch.Tensor)

    if y.shape == pred_y.shape:
        return y
    assert y.dtype == torch.long

    nspatial = len(pred_y.shape) - 2
    grad = torch.zeros_like(pred_y)
    y = y.reshape(-1, 1, *((1,) * nspatial)).expand_as(grad)
    grad.scatter_(1, y, 1.)
    return grad


def attach_debug_probes(model, debug=False):
    r"""
    Returns an :class:`collections.OrderedDict` of :class:`Probe` objects for
    all modules in the model if :attr:`debug` is ``True``; otherwise, returns
    ``None``.

    Args:
        model (:class:`torch.nn.Module`): a model.
        debug (bool, optional): if True, return an OrderedDict of Probe objects
            for all modules in the model; otherwise returns ``None``.
            Default: ``False``.

    Returns:
        :class:`collections.OrderedDict`: dict of :class:`Probe` objects for
            all modules in the model.
    """
    if not debug:
        return None

    debug_probes = OrderedDict()
    for module_name, module in model.named_modules():
        debug_probe_target = "input" if module_name == "" else "output"
        debug_probes[module_name] = Probe(
            module, target=debug_probe_target)
    return debug_probes


def resize_saliency(tensor, saliency, size, mode):
    """Resize a saliency map.

    Args:
        tensor (:class:`torch.Tensor`): reference tensor.
        saliency (:class:`torch.Tensor`): saliency map.
        size (bool or tuple of int): if a tuple (i.e., (width, height),
            resize :attr:`saliency` to :attr:`size`. If True, resize
            :attr:`saliency: to the shape of :attr:`tensor`; otherwise,
            return :attr:`saliency` unchanged.
        mode (str): mode for :func:`torch.nn.functional.interpolate`.

    Returns:
        :class:`torch.Tensor`: Resized saliency map.
    """
    # print(size)
    if size is not False:
        if size is True:
            size = tensor.shape[2:]
        elif isinstance(size, tuple) or isinstance(size, list):
            # width, height -> height, width
            size = size[::-1]
        else:
            assert False, "resize must be True, False or a tuple."
        saliency = torch.nn.functional.interpolate(
            saliency, size, mode=mode, align_corners=False)
    # print(saliency.size())
    return saliency


def saliency(model,
             input,
             target,
             saliency_layer='',
             resize=False,
             resize_mode='bilinear',
             smooth=0,
             context_builder=NullContext,
             gradient_to_saliency=gradient_to_grad_cam_saliency,
             get_backward_gradient=get_backward_gradient,
             debug=False):
    """Apply a backprop-based attribution method to an image.

    The saliency method is specified by a suitable context factory
    :attr:`context_builder`. This context is used to modify the backpropagation
    algorithm to match a given visualization method. This:

    Args:
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        target (int or :class:`torch.Tensor`): target label(s).
        saliency_layer (str or :class:`torch.nn.Module`, optional): name of the
            saliency layer (str) or the layer itself (:class:`torch.nn.Module`)
            in the model at which to visualize. Default: ``''`` (visualize
            at input).
        resize (bool or tuple, optional): if True, upsample saliency map to the
            same size as :attr:`input`. It is also possible to specify a pair
            (width, height) for a different size. Default: ``False``.
        resize_mode (str, optional): upsampling method to use. Default:
            ``'bilinear'``.
        smooth (float, optional): amount of Gaussian smoothing to apply to the
            saliency map. Default: ``0``.
        context_builder (type, optional): type of context to use. Default:
            :class:`NullContext`.
        gradient_to_saliency (function, optional): function that converts the
            pseudo-gradient signal to a saliency map. Default:
            :func:`gradient_to_saliency`.
        get_backward_gradient (function, optional): function that generates
            gradient tensor to backpropagate. Default:
            :func:`get_backward_gradient`.
        debug (bool, optional): if True, also return an
            :class:`collections.OrderedDict` of :class:`Probe` objects for
            all modules in the model. Default: ``False``.

    Returns:
        :class:`torch.Tensor` or tuple: If :attr:`debug` is False, returns a
        :class:`torch.Tensor` saliency map at :attr:`saliency_layer`.
        Otherwise, returns a tuple of a :class:`torch.Tensor` saliency map
        at :attr:`saliency_layer` and an :class:`collections.OrderedDict`
        of :class:`Probe` objects for all modules in the model.
    """
    # Clear any existing gradient.
    # print(saliency_layer)
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradients for model parameters.
    orig_requires_grad = {}
    for name, param in model.named_parameters():
        orig_requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Set model to eval mode.
    if model.training:
        orig_is_training = True
        model.eval()
    else:
        orig_is_training = False

    # Attach debug probes to every module.
    debug_probes = attach_debug_probes(model, debug=debug)

    # Attach a probe to the saliency layer.
    probe_target = 'input' if saliency_layer == '' else 'output'
    saliency_layer = get_module(model, saliency_layer)
    assert saliency_layer is not None, 'We could not find the saliency layer'
    probe = Probe(saliency_layer, target=probe_target)

    # Do a forward and backward pass.
    with context_builder():
        output = model(input)
        backward_gradient = get_backward_gradient(output, target)
        # print(backward_gradient)
        output.backward(backward_gradient)

    # Get saliency map from gradient.
    # print(probe.data[0].size())
    saliency_map = gradient_to_saliency(probe.data[0])

    # Resize saliency map.
    saliency_map = resize_saliency(input,
                                   saliency_map,
                                   resize,
                                   mode=resize_mode)

    # Smooth saliency map.
    if smooth > 0:
        saliency_map = imsmooth(
            saliency_map,
            sigma=smooth * max(saliency_map.shape[2:]),
            padding_mode='replicate'
        )

    # Remove probe.
    probe.remove()

    # Restore gradient saving for model parameters.
    for name, param in model.named_parameters():
        param.requires_grad_(orig_requires_grad[name])

    # Restore model's original mode.
    if orig_is_training:
        model.train()

    if debug:
        return saliency_map, debug_probes
    else:
        return saliency_map


def grad_cam(*args,
             saliency_layer,
             **kwargs):
    r"""Grad-CAM method.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the Grad-CAM method, and supports the
    same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_grad_cam_saliency,
                    **kwargs, )

def pil_to_tensor(pil_image):
    pil_image = np.array(pil_image)
    if len(pil_image.shape) == 2:
        pil_image = pil_image[:, :, None]
    return torch.tensor(pil_image, dtype=torch.float32).permute(2, 0, 1) / 255

def imread(img, as_pil=False, resize=None, to_rgb=False):
    if to_rgb:
        img = img.convert('RGB')
    if resize is not None:
        if not isinstance(resize, tuple) and not isinstance(resize, list):
            scale = float(resize) / float(min(img.size[0], img.size[1]))
            resize = [round(scale * h) for h in img.size]
        if resize != img.size:
            img = img.resize(resize, Image.ANTIALIAS)
    if as_pil:
        return img
    return pil_to_tensor(img)

def translate_alpha_to_x(alpha, x_input, x_baseline):
  """Translates alpha to the point coordinates within straight-line interval.

   Args:
    alpha: the relative location of the point between x_baseline and x_input.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.

  Returns:
    The coordinates of the point within [x_baseline, x_input] interval
    that correspond to the given value of alpha.
  """
  assert 0 <= alpha <= 1.0
  return x_baseline + (x_input - x_baseline) * alpha

def translate_x_to_alpha(x, x_input, x_baseline):
  """Translates a point on straight-line path to its corresponding alpha value.

  Args:
    x: the point on the straight-line path.
    x_input: the end point of the straight-line path.
    x_baseline: the start point of the straight-line path.

  Returns:
    The alpha value in range [0, 1] that shows the relative location of the
    point between x_baseline and x_input.
  """
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.where(x_input - x_baseline != 0,
                    (x - x_baseline) / (x_input - x_baseline), np.nan)


def l1_distance(x1, x2):
  """Returns L1 distance between two points."""
  return np.abs(x1 - x2).sum()

def CAM(x_input, scale, class_id):
    img = Image.fromarray(np.uint8(x_input))
    image_torch = imread(img, resize=(scale, scale), to_rgb=True).unsqueeze(0)
    # print(baseline + ((image_torch[0]*255.) - baseline))
    image_torch = image_torch.to(device)
    input = image_torch * 255.
    ex_stack = grad_cam(model, input.to(device),
                        saliency_layer='layer4.2', target=class_id)
    ex = ((ex_stack.cpu()).detach()).numpy()[0]
    exmap = cv2.resize(ex[0], (scale, scale))
    return exmap

def GIG_CAM(img_path, scale, class_id):
    image_PIL = Image.open(img_path)
    x_input = image_PIL.resize((scale, scale))
    attr = torch.zeros((scale, scale))
    x_input = np.asarray(x_input, dtype=np.float64)
    x_baseline = np.zeros(x_input.shape)
    x_baseline = np.asarray(x_baseline, dtype=np.float64)
    x = x_baseline.copy()
    l1_total = l1_distance(x_input, x_baseline)
    max_dist = 0.02
    classes = torchray.benchmark.datasets.VOC_CLASSES

    # print(f'Chosen class_id : {classes[class_id]}')

    x_path = []
    for step in range(30):
        saliency_actual = CAM(x, scale=scale, class_id=class_id)
        saliency_actual1 = torch.from_numpy(saliency_actual)
        saliency_actual1 = torch.clamp(saliency_actual1, min=0)
        saliency_map = saliency_actual.copy()
        alpha = (step + 1.0) / 30
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = translate_alpha_to_x(alpha_min, x_input, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, x_input, x_baseline)
        l1_target = l1_total * (1 - (step + 1) / 30)
        gamma = np.inf
        while gamma > 1.0:
            x_old = x.copy()
            x_alpha = translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha[np.isnan(x_alpha)] = alpha_max
            # All features that fell behind the [alpha_min, alpha_max] interval in
            # terms of alpha, should be assigned the x_min values.
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            # Calculate current L1 distance from the input.
            l1_current = l1_distance(x, x_input)
            # If the current L1 distance is close enough to the desired one then
            # update the attribution and proceed to the next step.
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += saliency_actual1
                break
            saliency_map[x.sum(axis=2) == x_max.sum(axis=2)] = np.inf
            threshold = np.quantile(np.abs(saliency_map), 0.25, method='lower')
            s = np.logical_and(np.abs(saliency_map) <= threshold, saliency_map != np.inf)
            l1_s = (np.abs(x - x_max) * np.transpose([s, s, s], (1, 2, 0))).sum()
            if l1_s > 0:
                gamma = (l1_current - l1_target) / l1_s
            else:
                gamma = np.inf

            if gamma > 1.0:
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]
            # Update attribution to reflect changes in `x`.

            attr += saliency_actual1
    return attr


VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
# classes = models.ResNet50_Weights.IMAGENET1K_V2.value.meta["categories"]
colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
def voc_label_indices(colormap):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]



scale_show = 324
img_path = 'cat_dog.png'
image_PIL = Image.open(img_path)
image_torch = imread(image_PIL, resize=(224, 224), to_rgb=True).unsqueeze(0)
predictions = model((image_torch * 255.).to(device))
predictions = torch.max(predictions, dim=1)[1]
predictions = predictions.cpu().data.numpy()[0]
# print(predictions)
class_id = predictions

result = GIG_CAM(img_path=img_path, scale=224, class_id=class_id)
result_224 = result.numpy()
result_224 = cv2.resize(result_224, (324, 324), interpolation=cv2.INTER_LINEAR)
result_fusion = result_224
result_add = result_224
for scale in list(range(324, 900, 100)):
    print('scale:' + str(scale))
    attr = GIG_CAM(img_path=img_path, scale=scale, class_id=class_id)
    pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)

    attr2 = pool(attr.reshape(1, scale, scale))
    attr2 = attr2[0].numpy()
    attr = attr.numpy()
    attr2 = cv2.resize(attr2, (scale, scale), interpolation=cv2.INTER_LINEAR)
    attr_norm = np.divide(attr, attr2, out=np.zeros_like(attr, dtype=np.float64), where=attr2 != 0)
    result_fusion = cv2.resize(result_fusion, (scale, scale), interpolation=cv2.INTER_LINEAR)
    result_fusion = attr_norm * result_fusion
    result_add = cv2.resize(result_add, (scale, scale), interpolation=cv2.INTER_LINEAR)
    result_add += result_fusion
data1 = 0.2  # 省略
result_add = (result_add - np.min(result_add)) / (np.max(result_add) - np.min(result_add))
mask_add = cv2.resize(result_add, (scale_show, scale_show), interpolation=cv2.INTER_LINEAR)
mask_add[mask_add < data1] = 0
plt.imshow(image_PIL.resize((scale_show, scale_show)))
plt.imshow(mask_add, cmap='jet', alpha=0.5);
plt.show()