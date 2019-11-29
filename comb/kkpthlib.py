import numpy as np
import torch
from scipy import linalg
from scipy.stats import truncnorm
from scipy.misc import factorial

def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros
    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize
    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype("float32")


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01
    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype("float32")


def np_truncated_normal(shape, random_state, scale=0.075):
    """
    Builds a numpy variable filled with truncated normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.075)
        default of 0.075
    Returns
    -------
    initialized_normal, array-like
        Array-like of truncated normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape

    sigma = scale
    lower = -2 * sigma
    upper = 2 * sigma
    mu = 0
    N = np.prod(shp)
    samples = truncnorm.rvs(
              (lower - mu) / float(sigma), (upper - mu) / float(sigma),
              loc=mu, scale=sigma, size=N, random_state=random_state)
    return samples.reshape(shp).astype("float32")


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale
    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype("float32")


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / float(kern_sum))  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_glorot_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1. * sqrt(6 / (n_in + n_out)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    """
    shp = shape
    kern_sum = sum(shp)
    bound = scale * np.sqrt(6. / float(kern_sum))
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.
    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prod(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims, name=""):
    logger.info("Initializing {} with {} init".format(name, "zero"))
    #return [np.random.randn(dim,).astype("float32") for dim in bias_dims]
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default", name=""):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    fs = [scale] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            ff[i] = np_ortho
            fs[i] = 1.
            '''
            if in_dim == out_dim:
                logger.info("Initializing {} with {} init".format(name, "ortho"))
                ff[i] = np_ortho
                fs[i] = 1.
            else:
                logger.info("Initializing {} with {} init".format(name, "variance_scaled_uniform"))
                ff[i] = np_variance_scaled_uniform
                fs[i] = 1.
            '''
        elif init == "ortho":
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            if in_dim != out_dim:
                raise ValueError("Unable to use ortho init for non-square matrices!")
            ff[i] = np_ortho
            fs[i] = 1.
        elif init == "glorot_uniform":
            logger.info("Initializing {} with {} init".format(name, "glorot_uniform"))
            ff[i] = np_glorot_uniform
        elif init == "normal":
            logger.info("Initializing {} with {} init".format(name, "normal"))
            ff[i] = np_normal
            fs[i] = 0.01
        elif init == "truncated_normal":
            logger.info("Initializing {} with {} init".format(name, "truncated_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 0.075
        elif init == "embedding_normal":
            logger.info("Initializing {} with {} init".format(name, "embedding_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 1. / np.sqrt(out_dim)
        else:
            raise ValueError("Unknown init type %s" % init)

    ws = []
    for i, out_dim in enumerate(out_dims):
        if fs[i] == "default":
            wi = ff[i]((in_dim, out_dim), random_state)
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
        else:
            wi = ff[i]((in_dim, out_dim), random_state, scale=fs[i])
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
    return ws

from scipy.stats import truncnorm
import sys
import uuid
import logging
from collections import OrderedDict
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there for html logger
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_logger():
    return logger

sys.setrecursionlimit(40000)

# Storage of internal shared
_lib_shared_params = OrderedDict()
has_warned = {}

def _shape(arr):
    return tuple(arr.shape)


def _ndim(arr):
    return len(_shape(arr))

def _get_name():
    return str(uuid.uuid4())


def _get_shared(name):
    if name in _lib_shared_params.keys():
        if name not in has_warned:
            logger.info("Found name %s in shared parameters" % name)
            has_warned[name] = True
        return _lib_shared_params[name]
    else:
        raise NameError("Name not found in shared params!")


def _check_shared(name):
    return name in _lib_shared_params.keys()


def _set_shared(name, variable):
    if name in _lib_shared_params.keys():
        raise ValueError("Trying to set key %s which already exists!" % name)
    _lib_shared_params[name] = variable

weight_norm_default = False
def get_weight_norm_default():
    return weight_norm_default

strict_mode_default = False
def get_strict_mode_default():
    return strict_mode_default


device_default = "cpu"
def get_device_default():
    return device_default

def set_device_default(device):
    global device_default
    device_default = device


dtype_default = "float32"
def get_dtype_default():
    return dtype_default

def set_dtype_default(dtype):
    global dtype_default
    dtype_default = dtype


def sigmoid(x):
    return torch.sigmoid(x)


def Sigmoid(x):
    return sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def Tanh(x):
    return tanh(x)


def relu(x):
    return torch.nn.functional.relu(x)


def ReLU(x):
    return relu(x)


def make_tensor(arr, dtype, device, requires_grad=True):
    if device == "default":
        device = get_device_default()
    else:
        device = device

    if dtype == "default":
        dtype = get_dtype_default()

    if dtype == "float32":
        tensor = torch.FloatTensor(arr, device=device)
    elif dtype == "float64":
        tensor = torch.DoubleTensor(arr, device=device)
    else:
        raise ValueError("Not yet implemented for dtype {}".format(dtype))
    if not requires_grad:
        tensor = tensor.requires_grad_(False)
    return tensor

def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    a_tup = _shape(a)
    b_tup = _shape(b)
    if len(a_tup) == 2 and len(b_tup) == 2:
        return torch.matmul(a, b)
    elif len(a_tup) == 3 and len(b_tup) == 2:
        # more generic, supports multiple -1 axes
        return torch.einsum("ijk,kl->ijl", a, b)
        #a_i = tf.reshape(a, [-1, a_tup[-1]])
        #a_n = tf.matmul(a_i, b)
        #a_nf = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
        #return a_nf
    else:
        raise ValueError("Shapes for arguments to dot() are {} and {}, not supported!".format(a_tup, b_tup))

_scan_infos = {}
def scan(fn, sequences, outputs_info):
    nonepos = [n for n, o in enumerate(outputs_info) if o is None]
    nonnone = [o for o in outputs_info if o is not None]
    sequences_and_nonnone = sequences + nonnone
    sliced = [s[0] for s in sequences] + nonnone
    lu = hash(fn)
    global _scan_infos
    if lu not in _scan_infos:
        inf_ret = fn(*sliced)
        _scan_infos[lu] = inf_ret
    else:
        inf_ret = scan_infos[lu]
    if len(outputs_info) < len(inf_ret):
        raise ValueError("More outputs from `fn` than elements in outputs_info. Expected {} outs, given outputs_info of length {}, but `fn` returns {}. Pass None in outputs_info for returns which don't accumulate".format(len(outputs_info), len(outputs_info), len(inf_ret)))
    initializers = []
    for n in range(len(outputs_info)):
        if outputs_info[n] is not None:
            initializers.append(outputs_info[n])
        else:
            initializers.append(0. * inf_ret[n])

    def wrapwrap(nonepos, initializers):
        type_class = "list" if isinstance(initializers, list) else "tuple"
        def fnwrap(accs, inps):
            inps_then_accs = inps + [a for n, a in enumerate(accs) if n not in nonepos]
            fn_rets = fn(*inps_then_accs)
            return [fr for fr in fn_rets]
        return fnwrap

    this_fn = wrapwrap(nonepos, initializers)
    def _scan(lclfn, seqs, inits):
        all_r = [[] for i in range(len(inits))]
        last_out = inits
        for i in range(len(seqs[0])):
            ri = lclfn(last_out, [seqs[n][i] for n in range(len(seqs))])
            last_out = ri
            if not hasattr(ri, "__len__"):
                ri = [ri]
            else:
                [all_r[j].append(ri[j]) for j in range(len(ri))]
        return all_r

    r = _scan(this_fn, sequences, initializers)
    return [torch.stack(rj) for rj in r]


class Embedding(torch.nn.Module):
    def __init__(self,
                 n_symbols,
                 output_dim,
                 random_state=None,
                 init="embedding_normal",
                 scale=1.,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        """
        Last dimension of indices tensor must be 1!!!!
        """
        super(Embedding, self).__init__()

        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass random_state argument to Embedding")

        name_w = name + "_embedding_w"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

        if init != "embedding_normal":
            raise ValueError("Currently unsupported init type {}".format(init))

        try:
            vectors = _get_shared(name_w)
        except NameError:
            vectors_weight, = make_numpy_weights(n_symbols, [output_dim],
                                                 random_state, init=init,
                                                 scale=scale, name=name_w)
            vectors = make_tensor(vectors_weight, dtype=dtype, device=device)
            #vectors = torch.from_numpy(vectors_weight).to(lcl_device)
            _set_shared(name_w, vectors)
        self.vectors = vectors

        th_embed = torch.nn.Embedding(n_symbols, output_dim)
        th_embed.weight.data.copy_(vectors)
        self.th_embed = th_embed

    def forward(self,
                indices):
        ii = indices.long()
        shp = _shape(ii)
        nd = _ndim(ii)
        if shp[-1] != 1:
            if nd < 3:
                logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
                ii = ii[..., None]
            else:
                raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

        shp = _shape(ii)
        nd = len(shp)
        # force 3d for consistency, then slice
        lu = self.th_embed(ii[..., 0])
        return lu, self.vectors


'''
def Embedding(indices, n_symbols, output_dim, random_state=None,
              init="embedding_normal", scale=1.,
              strict=None, name=None, dtype="default", device="default"):
    """
    Last dimension of indices tensor must be 1!!!!
    """
    shp = _shape(indices)

    if name is None:
        name = _get_name()

    if random_state is None:
        raise ValueError("Must pass random_state argument to Embedding")

    name_w = name + "_embedding_w"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

    if init != "embedding_normal":
        raise ValueError("Currently unsupported init type {}".format(init))

    try:
        vectors = _get_shared(name_w)
    except NameError:
        vectors_weight, = make_numpy_weights(n_symbols, [output_dim],
                                             random_state, init=init,
                                             scale=scale, name=name_w)
        vectors = make_tensor(vectors_weight, dtype=dtype, device=device)
        #vectors = torch.from_numpy(vectors_weight).to(lcl_device)
        _set_shared(name_w, vectors)

    th_embed = torch.nn.Embedding(n_symbols, output_dim)
    th_embed.weight.data.copy_(vectors)

    ii = indices.long()
    shp = _shape(ii)
    nd = _ndim(ii)
    if shp[-1] != 1:
        if nd < 3:
            logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
            ii = ii[..., None]
        else:
            raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

    shp = _shape(ii)
    nd = len(shp)
    # force 3d for consistency, then slice
    lu = th_embed(ii[..., 0])
    return lu, vectors
'''



class Linear(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 output_dim,
                 random_state=None,
                 name=None,
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 dropout_flag_prob_keep=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(Linear, self).__init__()
        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")
        input_dim = sum(list_of_input_dims)

        if name is None:
            name = _get_name()

        name_w = name + "_linear_w"
        name_b = name + "_linear_b"
        name_out = name + "_linear_out"

        if init is None or type(init) is str:
            #logger.info("Linear layer {} initialized using init {}".format(name, init))
            weight_values, = make_numpy_weights(input_dim, [output_dim],
                                                random_state=random_state,
                                                init=init, scale=scale, name=name_w)
        else:
            # rely on announcement from parent class
            weight_values=init[0]

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)
        self.weight = weight
        self.biases = None

        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([output_dim], name=name_b)
            else:
                b = init[1]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device)
                _set_shared(name_b, biases)
            self.biases = biases

    def forward(self,
                list_of_inputs,
                bias_offset=0.,
                dropout_flag_prob_keep=None):

        nd = _ndim(list_of_inputs[0])
        input_var = torch.cat(list_of_inputs, dim=nd - 1)
        if dropout_flag_prob_keep is not None:
            # no seed set here, it might not be repeatable
            input_var = torch.nn.functional.dropout(input_var, p=1. - dropout_flag_prob_keep, inplace=False)

        out = dot(input_var, self.weight)

        if self.biases is not None:
            out = out + self.biases
        return out

"""
def Linear(list_of_inputs, list_of_input_dims, output_dim, random_state=None,
           name=None, init=None, scale="default", biases=True, bias_offset=0.,
           dropout_flag_prob_keep=None, strict=None, dtype="default", device="default"):
    if random_state is None:
        raise ValueError("Must pass instance of np.random.RandomState!")
    nd = _ndim(list_of_inputs[0])
    input_var = torch.cat(list_of_inputs, dim=nd - 1)
    input_dim = sum(list_of_input_dims)

    if name is None:
        name = _get_name()

    name_w = name + "_linear_w"
    name_b = name + "_linear_b"
    name_out = name + "_linear_out"

    if init is None or type(init) is str:
        #logger.info("Linear layer {} initialized using init {}".format(name, init))
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale, name=name_w)
    else:
        # rely on announcement from parent class
        weight_values=init[0]


    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

    try:
        weight = _get_shared(name_w)
    except NameError:
        weight = make_tensor(weight_values, dtype=dtype, device=device)
        _set_shared(name_w, weight)

    if dropout_flag_prob_keep is not None:
        # no seed set here, it might not be repeatable
        input_var = torch.nn.functional.dropout(input_var, p=1. - dropout_flag_prob_keep, inplace=False)

    out = dot(input_var, weight)

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([output_dim], name=name_b)
        else:
            b = init[1]
        b = b + bias_offset
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = make_tensor(b, dtype=dtype, device=device)
            _set_shared(name_b, biases)
        out = out + biases
    return out
"""

class Conv2d(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 kernel_size=(3, 3),
                 dilation=[1, 1],
                 strides=[1, 1],
                 input_height_width_init_tuple=(1, 1),
                 border_mode="same",
                 custom_weight_mask=None,
                 init=None, scale="default",
                 biases=True, bias_offset=0.,
                 name=None, random_state=None, strict=None,
                 dtype="default", device="default"):
        super(Conv2d, self).__init__()
        if strides != [1, 1]:
            raise ValueError("Alternate strides not yet supported in conv2d")
        if dilation != [1, 1]:
            raise ValueError("Alternate dilation not yet supported in conv2d")
        # kernel is H, W
        # input assumption is N C H W
        if name is None:
            name = _get_name()

        if random_state is None:
            raise ValueError("Must pass instance of np.random.RandomState!")

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            raise ValueError("Changing dilation not yet supported")

        input_channels = sum(list_of_input_dims)
        #input_height, input_width = input_height_width_tuple
        # these numbers don't matter
        input_height, input_width = input_height_width_init_tuple

        if type(name) is str:
            name_w = name + "_conv2d_w"
            name_b = name + "_conv2d_b"
            name_out = name + "_conv2d_out"
            name_mask = name + "_conv2d_mask"

        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_w in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_w))

            if name_b in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_b))

        if init is None or type(init) is str:
            weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                                [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                                init=init,
                                                scale=scale,
                                                random_state=random_state, name=name_w)
        else:
            weight_values = init[0]
            name_w = name[0]
        weight_values = weight_values.transpose(3, 2, 0, 1)
        #weight_values = weight_values[::-1, ::-1].copy()

        try:
            weight = _get_shared(name_w)
        except NameError:
            weight = make_tensor(weight_values, dtype=dtype, device=device)
            _set_shared(name_w, weight)

        self.weight = weight

        if custom_weight_mask is not None:
            """
            try:
                mask = _get_shared(name_mask)
            except NameError:
                mask = tf.Variable(custom_weight_mask, trainable=False, name=name_mask)
                _set_shared(name_mask, mask)
            """
            raise ValueError("custom_weight_mask not yet implemented in conv")
            weight = tf.constant(custom_weight_mask) * weight

        # need to custom handle SAME and VALID
        # rip
        # NCHW input, weights are out_chan, in_chan, H, W
        if biases:
            if (init is None) or (type(init) is str):
                b, = make_numpy_biases([num_feature_maps], name=name_b)
            else:
                b = init[1]
                name_b = name[1]
                name_out = name[2]
            b = b + bias_offset
            try:
                biases = _get_shared(name_b)
            except NameError:
                biases = make_tensor(b, dtype=dtype, device=device)
                _set_shared(name_b, biases)
        self.biases = biases
        self.strides = strides
        self.dilation = dilation
        self.input_channels = input_channels
        # these numbers don't matter
        self.input_height = input_height
        self.input_width = input_width
        self.border_mode = border_mode
        self.kernel_size = kernel_size

    def forward(self,
                list_of_inputs):
        dilation = self.dilation
        strides = self.strides
        input_channels = self.input_channels
        input_height = self.input_height
        input_width = self.input_width
        border_mode = self.border_mode
        weight = self.weight
        biases = self.biases
        kernel_size = self.kernel_size

        if strides != [1, 1]:
            raise ValueError("Alternate strides not yet supported in conv2d")
        if dilation != [1, 1]:
            raise ValueError("Alternate dilation not yet supported in conv2d")
        # kernel is H, W
        # input assumption is N C H W

        if strides != [1, 1]:
            if hasattr(strides, "__len__") and len(strides) == 2:
                pass
            else:
                try:
                    int(strides)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Changing strides by non-int not yet supported")

        if dilation != [1, 1]:
            raise ValueError("Changing dilation not yet supported")

        input_t = torch.cat(list_of_inputs, dim=-1)

        if border_mode == "same":
            pad = "same"
        elif border_mode == "valid":
            pad = "valid"
        else:
            pad = border_mode
            if hasattr(pad, "__len__") and len(pad) == 2:
                pass
            else:
                try:
                    int(pad)
                    strides = [int(strides), int(strides)]
                except:
                    raise ValueError("Pad must be integer, tuple of integer (hpad, wpad), or string 'same', 'valid'")

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim(in_dim, padding, ks, stride, dilation):
            if isinstance(padding, int) or isinstance(padding, tuple):
                return conv_outdim_general(in_dim, padding, ks, stride, dilation)
            elif isinstance(padding, str):
                assert padding in ['same', 'valid']
                if padding == 'same':
                    return conv_outdim_samepad(in_dim, stride)
                else:
                    return conv_outdim_general(in_dim, 0, ks, stride, dilation)
            else:
                raise TypeError('Padding can be int/tuple or str=same/valid')

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_general(in_dim, padding, ks, stride, dilation=1):
            # See https://arxiv.org/pdf/1603.07285.pdf, eq (15)
            return ((in_dim + 2 * padding - ks - (ks - 1) * (dilation - 1)) // stride) + 1

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def conv_outdim_samepad(in_dim, stride):
            return (in_dim + stride - 1) // stride

        # https://github.com/pytorch/pytorch/issues/3867
        # credit to @mirceamironenco
        def pad_same(in_dim, ks, stride, dilation=1):
            """
            References:
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
                  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
            """
            assert stride > 0
            assert dilation >= 1
            effective_ks = (ks - 1) * dilation + 1
            out_dim = (in_dim + stride - 1) // stride
            p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

            padding_before = p // 2
            padding_after = p - padding_before
            return padding_before, padding_after


        if pad == "same":
            ph = pad_same(input_t.shape[-2], kernel_size[0], strides[-2], dilation[-2])[0]
            pw = pad_same(input_t.shape[-1], kernel_size[1], strides[-1], dilation[-1])[0]
        elif pad == "valid":
            raise ValueError("valid pad NYI")
            from IPython import embed; embed(); raise ValueError()

        # NCHW input, weights are out_chan, in_chan, H, W
        out = torch.nn.functional.conv2d(input_t, weight, stride=strides, dilation=dilation, padding=(ph, pw), bias=biases)
        return out


'''
def Conv2d(list_of_inputs, list_of_input_dims, num_feature_maps,
           kernel_size=(3, 3),
           dilation=[1, 1],
           strides=[1, 1],
           border_mode="same",
           custom_weight_mask=None,
           init=None, scale="default",
           biases=True, bias_offset=0.,
           name=None, random_state=None, strict=None,
           dtype="default", device="default"):
    if strides != [1, 1]:
        raise ValueError("Alternate strides not yet supported in conv2d")
    if dilation != [1, 1]:
        raise ValueError("Alternate dilation not yet supported in conv2d")
    # kernel is H, W
    # input assumption is N C H W
    if name is None:
        name = _get_name()

    if random_state is None:
        raise ValueError("Must pass instance of np.random.RandomState!")

    if strides != [1, 1]:
        if hasattr(strides, "__len__") and len(strides) == 2:
            pass
        else:
            try:
                int(strides)
                strides = [int(strides), int(strides)]
            except:
                raise ValueError("Changing strides by non-int not yet supported")

    if dilation != [1, 1]:
        raise ValueError("Changing dilation not yet supported")

    input_t = torch.cat(list_of_inputs, dim=-1)
    input_channels = sum(list_of_input_dims)
    input_height = _shape(input_t)[2]
    input_width = _shape(input_t)[3]

    if type(name) is str:
        name_w = name + "_conv2d_w"
        name_b = name + "_conv2d_b"
        name_out = name + "_conv2d_out"
        name_mask = name + "_conv2d_mask"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

    if init is None or type(init) is str:
        weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                            [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                            init=init,
                                            scale=scale,
                                            random_state=random_state, name=name_w)
    else:
        weight_values = init[0]
        name_w = name[0]
    weight_values = weight_values.transpose(3, 2, 0, 1)
    #weight_values = weight_values[::-1, ::-1].copy()

    try:
        weight = _get_shared(name_w)
    except NameError:
        weight = make_tensor(weight_values, dtype=dtype, device=device)
        _set_shared(name_w, weight)

    if custom_weight_mask is not None:
        """
        try:
            mask = _get_shared(name_mask)
        except NameError:
            mask = tf.Variable(custom_weight_mask, trainable=False, name=name_mask)
            _set_shared(name_mask, mask)
        """
        raise ValueError("custom_weight_mask not yet implemented in conv")
        weight = tf.constant(custom_weight_mask) * weight

    # need to custom handle SAME and VALID
    # rip

    if border_mode == "same":
        pad = "same"
    elif border_mode == "valid":
        pad = "valid"
    else:
        pad = border_mode
        if hasattr(pad, "__len__") and len(pad) == 2:
            pass
        else:
            try:
                int(pad)
                strides = [int(strides), int(strides)]
            except:
                raise ValueError("Pad must be integer, tuple of integer (hpad, wpad), or string 'same', 'valid'")

    # https://github.com/pytorch/pytorch/issues/3867
    # credit to @mirceamironenco
    def conv_outdim(in_dim, padding, ks, stride, dilation):
        if isinstance(padding, int) or isinstance(padding, tuple):
            return conv_outdim_general(in_dim, padding, ks, stride, dilation)
        elif isinstance(padding, str):
            assert padding in ['same', 'valid']
            if padding == 'same':
                return conv_outdim_samepad(in_dim, stride)
            else:
                return conv_outdim_general(in_dim, 0, ks, stride, dilation)
        else:
            raise TypeError('Padding can be int/tuple or str=same/valid')

    # https://github.com/pytorch/pytorch/issues/3867
    # credit to @mirceamironenco
    def conv_outdim_general(in_dim, padding, ks, stride, dilation=1):
        # See https://arxiv.org/pdf/1603.07285.pdf, eq (15)
        return ((in_dim + 2 * padding - ks - (ks - 1) * (dilation - 1)) // stride) + 1

    # https://github.com/pytorch/pytorch/issues/3867
    # credit to @mirceamironenco
    def conv_outdim_samepad(in_dim, stride):
        return (in_dim + stride - 1) // stride

    # https://github.com/pytorch/pytorch/issues/3867
    # credit to @mirceamironenco
    def pad_same(in_dim, ks, stride, dilation=1):
        """
        References:
              https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
              https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
        """
        assert stride > 0
        assert dilation >= 1
        effective_ks = (ks - 1) * dilation + 1
        out_dim = (in_dim + stride - 1) // stride
        p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

        padding_before = p // 2
        padding_after = p - padding_before
        return padding_before, padding_after


    if pad == "same":
        ph = pad_same(input_t.shape[-2], kernel_size[0], strides[-2], dilation[-2])[0]
        pw = pad_same(input_t.shape[-1], kernel_size[1], strides[-1], dilation[-1])[0]
    elif pad == "valid":
        raise ValueError("valid pad NYI")
        from IPython import embed; embed(); raise ValueError()

    # NCHW input, weights are out_chan, in_chan, H, W
    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([num_feature_maps], name=name_b)
        else:
            b = init[1]
            name_b = name[1]
            name_out = name[2]
        b = b + bias_offset
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = make_tensor(b, dtype=dtype, device=device)
            _set_shared(name_b, biases)
    out = torch.nn.functional.conv2d(input_t, weight, stride=strides, dilation=dilation, padding=(ph, pw), bias=biases)
    return out
'''

class BatchNorm2d(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 train_test_flag,
                 gamma_init=1., beta_init=0.,
                 decay=0.9,
                 eps=1E-3,
                 strict=None,
                 name=None,
                 dtype="default",
                 device="default"):
        super(BatchNorm2d, self).__init__()
        # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        # NCHW convention
        if name is None:
            name = _get_name()

        name_scale = name + "_batchnorm_s"
        name_beta = name + "_batchnorm_b"
        name_out = name + "_batchnorm_out"
        if strict is None:
            strict = get_strict_mode_default()

        if strict:
            cur_defs = get_params_dict()
            if name_scale in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_scale))

            if name_beta in cur_defs:
                raise ValueError("Name {} already created in params dict!".format(name_beta))

        try:
            scale = _get_shared(name_scale)
        except NameError:
            scale_values = gamma_init * np.ones((input_dim,))
            scale = make_tensor(scale_values, dtype=dtype, device=device)
            _set_shared(name_scale, scale)

        try:
            beta = _get_shared(name_beta)
        except NameError:
            # init with ones? it's what I did in TF
            beta_values = beta_init * np.ones((input_dim,))
            beta = make_tensor(beta_values, dtype=dtype, device=device)
            _set_shared(name_beta, beta)

        self.beta = beta
        self.scale = scale
        self.decay = decay
        self.eps = eps
        self.dtype = dtype
        self.device = device

    def forward(self, input_tensor, train_test_flag):
        # 0 train, 1 test
        # https://stackoverflow.com/questions/44887446/pytorch-nn-functional-batch-norm-for-2d-input
        scale = self.scale
        beta = self.beta
        eps = self.eps
        decay = self.decay
        dtype = self.dtype
        device = self.device

        pop_mean = make_tensor(np.zeros((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)
        pop_var = make_tensor(np.ones((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)

        shp = _shape(input_tensor)
        def left():
            return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, weight=scale, bias=beta, momentum=1. - decay, eps=eps, training=True)

        def right():
            return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, training=False, weight=scale, bias=beta, eps=eps)

        if train_test_flag <= 0.5:
            out = left()
        else:
            out = right()
        return out

'''
def BatchNorm2d(input_tensor, train_test_flag,
                gamma_init=1., beta_init=0.,
                decay=0.9,
                eps=1E-3,
                strict=None,
                name=None,
                dtype="default",
                device="default"):
    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    # NCHW convention
    if name is None:
        name = _get_name()

    name_scale = name + "_batchnorm_s"
    name_beta = name + "_batchnorm_b"
    name_out = name + "_batchnorm_out"
    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_scale in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_scale))

        if name_beta in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_beta))

    try:
        scale = _get_shared(name_scale)
    except NameError:
        scale_values = gamma_init * np.ones((input_tensor.shape[1],))
        scale = make_tensor(scale_values, dtype=dtype, device=device)
        _set_shared(name_scale, scale)

    try:
        beta = _get_shared(name_beta)
    except NameError:
        # init with ones? it's what I did in TF
        beta_values = beta_init * np.ones((input_tensor.shape[1],))
        beta = make_tensor(beta_values, dtype=dtype, device=device)
        _set_shared(name_beta, beta)

    # https://stackoverflow.com/questions/44887446/pytorch-nn-functional-batch-norm-for-2d-input
    pop_mean = make_tensor(np.zeros((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)
    pop_var = make_tensor(np.ones((input_tensor.shape[1],)), dtype=dtype, device=device, requires_grad=False)

    shp = _shape(input_tensor)
    def left():
        return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, weight=scale, bias=beta, momentum=1. - decay, eps=eps, training=True)

    def right():
        return torch.nn.functional.batch_norm(input_tensor, pop_mean, pop_var, training=False, weight=scale, bias=beta, eps=eps)

    if train_test_flag <= 0.5:
        out = left()
    else:
        out = right()
    return out
'''


class SequenceConv1dStack(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_feature_maps,
                 batch_norm_flag=None,
                 n_stacks=1,
                 residual=True,
                 activation="relu",
                 kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                 border_mode="same",
                 init=None,
                 scale="default",
                 biases=True,
                 bias_offset=0.,
                 name=None,
                 random_state=None,
                 strict=None,
                 dtype="default",
                 device="default"):
        super(SequenceConv1dStack, self).__init__()
        if name is None:
            name = _get_name()

        # assuming they come in as length, batch, features
        #tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        c = Conv2d(list_of_input_dims, len(kernel_sizes) * num_feature_maps,
                   kernel_size=(1, 1),
                   name=name + "_convpre", random_state=random_state,
                   border_mode=border_mode, init=init, scale=scale, biases=biases,
                   bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
        layers = torch.nn.ModuleList()
        layers.append(torch.nn.ModuleList([c,]))

        for ii in range(n_stacks):
            cs = torch.nn.ModuleList()
            for jj, ks in enumerate(kernel_sizes):
                c = Conv2d([len(kernel_sizes) * num_feature_maps], num_feature_maps,
                           kernel_size=ks,
                           name=name + "_conv{}_ks{}".format(ii, jj), random_state=random_state,
                           border_mode=border_mode, init=init, scale=scale, biases=biases,
                           bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
                cs.append(c)
            # cat along channel axis
            bn_l = BatchNorm2d(len(cs) * num_feature_maps, batch_norm_flag, name="bn_conv{}".format(ii), dtype=dtype, device=device)
            cs.append(bn_l)
            # ????
            #r_l = ReLU(bn_l)
            layers.append(cs)
            #prev_layer = prev_layer + r_l

        post = Conv2d([len(kernel_sizes) * num_feature_maps], num_feature_maps,
                       kernel_size=(1, 1),
                       name=name + "_convpost", random_state=random_state,
                       border_mode=border_mode, init=init, scale=scale, biases=biases,
                       bias_offset=bias_offset, strict=strict,
                       dtype=dtype, device=device)

        li = torch.nn.ModuleList()
        li.append(post)
        layers.append(li)
        self.layers = layers
        self.n_stacks = n_stacks
        self.kernel_sizes = kernel_sizes

    def forward(self,
                list_of_inputs,
                batch_norm_flag=None):

        # assuming they come in as length, batch, features
        tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
        # now N C H W, height of 1 (so laid out along width dim)
        pre = self.layers[0][0](tlist)
        n_stacks = self.n_stacks
        kernel_sizes = self.kernel_sizes
        prev_layer = pre
        for ii in range(n_stacks):
            cs = []
            for jj, ks in enumerate(kernel_sizes):
                # off by one to account for pre
                c = self.layers[ii + 1][jj]([prev_layer])
                cs.append(c)
            c_layer = torch.cat(cs, dim=1)
            # cat along channel axis
            # off by one to account for bn layer last
            bn_l = self.layers[ii + 1][jj + 1](c_layer, batch_norm_flag)
            r_l = ReLU(bn_l)
            prev_layer = prev_layer + r_l
        post = self.layers[ii + 2][0]([prev_layer])
        return post[:, :, 0].permute(2, 0, 1)

"""
def SequenceConv1dStack(list_of_inputs, list_of_input_dims, num_feature_maps,
                        batch_norm_flag,
                        n_stacks=1,
                        residual=True,
                        activation="relu",
                        kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                        border_mode="same",
                        init=None, scale="default",
                        biases=True, bias_offset=0.,
                        name=None, random_state=None, strict=None, dtype="default", device="default"):
    if name is None:
        name = _get_name()

    # assuming they come in as length, batch, features
    tlist = [li[:, None].permute((2, 3, 1, 0)) for li in list_of_inputs]
    # now N C H W, height of 1 (so laid out along width dim)

    c = Conv2d(tlist, list_of_input_dims, len(kernel_sizes) * num_feature_maps,
               kernel_size=(1, 1),
               name=name + "_convpre", random_state=random_state,
               border_mode=border_mode, init=init, scale=scale, biases=biases,
               bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
    prev_layer = c
    for ii in range(n_stacks):
        cs = []
        for jj, ks in enumerate(kernel_sizes):
            c = Conv2d([prev_layer], [len(kernel_sizes) * num_feature_maps], num_feature_maps,
                       kernel_size=ks,
                       name=name + "_conv{}_ks{}".format(ii, jj), random_state=random_state,
                       border_mode=border_mode, init=init, scale=scale, biases=biases,
                       bias_offset=bias_offset, strict=strict, dtype=dtype, device=device)
            cs.append(c)
        layer = torch.cat(cs, dim=1)
        # cat along channel axis
        bn_l = BatchNorm2d(layer, batch_norm_flag, name="bn_conv{}".format(ii), dtype=dtype, device=device)
        r_l = ReLU(bn_l)
        prev_layer = prev_layer + r_l
    post = Conv2d([prev_layer], [len(kernel_sizes) * num_feature_maps], num_feature_maps,
                   kernel_size=(1, 1),
                   name=name + "_convpost", random_state=random_state,
                   border_mode=border_mode, init=init, scale=scale, biases=biases,
                   bias_offset=bias_offset, strict=strict,
                   dtype=dtype, device=device)
    return post[:, :, 0].permute(2, 0, 1)
"""

class LSTMCell(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 input_mask=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None):
        super(LSTMCell, self).__init__()
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call
        if random_state is None:
            raise ValueError("Must pass random_state")

        if name is None:
            name = _get_name()

        input_dim = sum(list_of_input_dims)
        hidden_dim = 4 * num_units

        if init is None:
            inp_init = None
            h_init = None
            out_init = None
        elif init == "truncated_normal":
            inp_init = "truncated_normal"
            h_init = "truncated_normal"
            out_init = "truncated_normal"
        elif init == "glorot_uniform":
            inp_init = "glorot_uniform"
            h_init = "glorot_uniform"
            out_init = "glorot_uniform"
        elif init == "normal":
            inp_init = "normal"
            h_init = "normal"
            out_init = "normal"
        else:
            raise ValueError("Unknown init argument {}".format(init))

        name_proj = name + "_lstm_proj"
        name_w = name + "_lstm_proj_w"
        name_b = name + "_lstm_proj_b"
        comb_w_np, = make_numpy_weights(input_dim + num_units, [hidden_dim],
                                        random_state=random_state,
                                        init=inp_init, name=name_w)
        comb_b_np, = make_numpy_biases([hidden_dim], name=name_b)

        logger.info("LSTMCell {} input to hidden initialized using init {}".format(name, inp_init))
        logger.info("LSTMCell {} hidden to hidden initialized using init {}".format(name, h_init))

        lstm_proj_obj = Linear(list_of_input_dims + [hidden_dim],
                               hidden_dim,
                               random_state=random_state,
                               name=name_proj,
                               init=(comb_w_np, comb_b_np), strict=strict)

        if output_dim is not None:
            name_out = name + "_lstm_h_to_out",
            name_out_w = name + "_lstm_h_to_out_w",
            name_out_b = name + "_lstm_h_to_out_b",
            h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                                random_state=random_state,
                                                init=out_init, name=name_out_w)
            h_to_out_b_np, = make_numpy_biases([output_dim], name=name_out_b)
            h_to_out_obj = Linear([num_units], output_dim, random_state=random_state,
                              name=name_out,
                              init=(h_to_out_w_np, h_to_out_b_np), strict=strict)
            self.h_to_out_obj = h_to_out_obj
        self.lstm_proj_obj = lstm_proj_obj
        self.num_units = num_units
        self.input_dim = input_dim
        self.forget_bias = forget_bias

    def forward(self,
                list_of_inputs,
                previous_hidden, previous_cell,
                output_dim=None,
                input_mask=None,
                cell_dropout=None):
        # cell_dropout should be a value in [0., 1.], or None
        # output is the thing to use in following layers, state is a tuple that feeds into the next call

        input_dim = self.input_dim
        num_units = self.num_units
        forget_bias = self.forget_bias

        ph = previous_hidden
        pc = previous_cell

        lstm_proj = self.lstm_proj_obj(list_of_inputs + [ph])

        i = lstm_proj[..., :num_units]
        j = lstm_proj[..., num_units:2 * num_units]
        f = lstm_proj[..., 2 * num_units:3 * num_units]
        o = lstm_proj[..., 3 * num_units:]

        if cell_dropout is not None:
            pj = torch.nn.functional.dropout(tanh(j), cell_dropout)
        else:
            pj = tanh(j)

        c = sigmoid(f + forget_bias) * pc + sigmoid(i) * pj
        if input_mask is not None:
            c = input_mask[:, None] * c + (1. - input_mask[:, None]) * pc

        h = sigmoid(o) * tanh(c)
        if input_mask is not None:
            h = input_mask[:, None] * h + (1. - input_mask[:, None]) * h

        if output_dim is not None:
            h_to_out = self.h_to_out_obj([h])
            final_out = h_to_out
        else:
            final_out = h
        return final_out, (h, c)


'''
def LSTMCell(list_of_inputs, list_of_input_dims,
             previous_hidden, previous_cell,
             num_units,
             output_dim=None,
             input_mask=None,
             random_state=None,
             name=None, init=None, scale="default",
             forget_bias=1.,
             cell_dropout=None,
             strict=None):
    # cell_dropout should be a value in [0., 1.], or None
    # output is the thing to use in following layers, state is a tuple that feeds into the next call
    if random_state is None:
        raise ValueError("Must pass random_state")

    print("MAKE IT CLASSY LSTMCELL")
    from IPython import embed; embed(); raise ValueError()
    if name is None:
        name = _get_name()

    input_dim = sum(list_of_input_dims)
    hidden_dim = 4 * num_units

    if init is None:
        inp_init = None
        h_init = None
        out_init = None
    elif init == "truncated_normal":
        inp_init = "truncated_normal"
        h_init = "truncated_normal"
        out_init = "truncated_normal"
    elif init == "glorot_uniform":
        inp_init = "glorot_uniform"
        h_init = "glorot_uniform"
        out_init = "glorot_uniform"
    elif init == "normal":
        inp_init = "normal"
        h_init = "normal"
        out_init = "normal"
    else:
        raise ValueError("Unknown init argument {}".format(init))

    name_proj = name + "_lstm_proj"
    if _check_initted_cell(name_proj):
        l_func = _get_initted_cell(name_proj)
    else:
        name_w = name + "_lstm_proj_w"
        name_b = name + "_lstm_proj_b"
        comb_w_np, = make_numpy_weights(input_dim + num_units, [hidden_dim],
                                        random_state=random_state,
                                        init=inp_init, name=name_w)
        comb_b_np, = make_numpy_biases([hidden_dim], name=name_b)

        logger.info("LSTMCell {} input to hidden initialized using init {}".format(name, inp_init))
        logger.info("LSTMCell {} hidden to hidden initialized using init {}".format(name, h_init))

    lstm_proj = Linear(list_of_inputs + [ph], list_of_input_dims + [hidden_dim],
                       hidden_dim,
                       random_state=random_state,
                       name=name_proj,
                       init=(comb_w_np, comb_b_np), strict=strict)

    from IPython import embed; embed(); raise ValueError()
    i = lstm_proj[..., :num_units]
    j = lstm_proj[..., num_units:2 * num_units]
    f = lstm_proj[..., 2 * num_units:3 * num_units]
    o = lstm_proj[..., 3 * num_units:]

    if cell_dropout is not None:
        pj = torch.nn.functional.dropout(tanh(j), cell_dropout)
    else:
        pj = tanh(j)

    c = sigmoid(f + forget_bias) * pc + sigmoid(i) * pj
    if input_mask is not None:
        c = input_mask[:, None] * c + (1. - input_mask[:, None]) * pc

    h = sigmoid(o) * tanh(c)
    if input_mask is not None:
        h = input_mask[:, None] * h + (1. - input_mask[:, None]) * h

    print(h.shape)
    print(c.shape)

    if output_dim is not None:
        name_out = name + "_lstm_h_to_out",
        name_out_w = name + "_lstm_h_to_out_w",
        name_out_b = name + "_lstm_h_to_out_b",
        h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                            random_state=random_state,
                                            init=out_init, name=name_out_w)
        h_to_out_b_np, = make_numpy_biases([output_dim], name=name_out_b)
        h_to_out = Linear([h], [num_units], output_dim, random_state=random_state,
                          name=name_out,
                          init=(h_to_out_w_np, h_to_out_b_np), strict=strict)
        final_out = h_to_out
        #logger.info("LSTMCell {} hidden to output initialized using init {}".format(name, out_init))
    else:
        final_out = h
    return final_out, (h, c)
'''


'''
def BiLSTMLayer(list_of_inputs, list_of_input_dims,
                num_units,
                previous_forward_hidden=None, previous_forward_cell=None,
                previous_reverse_hidden=None, previous_reverse_cell=None,
                output_dim=None,
                input_mask=None,
                random_state=None,
                name=None, init=None, scale="default",
                forget_bias=1.,
                cell_dropout=None,
                strict=None):
    if input_mask is None:
        raise ValueError("No input mask currently unsupported")
    if name is None:
        name = _get_name()
    name = name + "_bidirlstm_layer"
    name_proj = name + "_proj"
    hidden_dim = 4 * num_units
    in_proj = Linear(list_of_inputs, list_of_input_dims,
                     hidden_dim,
                     random_state=random_state,
                     name=name_proj,
                     init=init, strict=strict)
    if previous_forward_hidden == None:
        h1_f_init = 0. * in_proj[0, :, :num_units]
    else:
        h1_f_init = previous_forward_hidden
    if previous_reverse_hidden == None:
        h1_b_init = 0. * in_proj[0, :, :num_units]
    else:
        h1_b_init = previous_reverse_hidden
    if previous_forward_cell == None:
        c1_f_init = 0. * in_proj[0, :, :num_units]
    else:
        c1_f_init = previous_forward_cell
    if previous_reverse_cell == None:
        c1_b_init = 0. * in_proj[0, :, :num_units]
    else:
        c1_b_init = previous_reverse_cell

    def step(inp_t, inp_mask_t,
             rev_inp_t, rev_inp_mask_t,
             h1_f_tm1, c1_f_tm1, h1_b_tm1, c1_b_tm1):
        output, s = LSTMCell([inp_t],
                             [hidden_dim],
                             h1_f_tm1, c1_f_tm1,
                             num_units,
                             input_mask=inp_mask_t,
                             random_state=random_state,
                             cell_dropout=cell_dropout,
                             name=name + "forward_rnn",
                             init=init)
        h1_f_t = s[0]
        c1_f_t = s[1]

        output, s = LSTMCell([rev_inp_t],
                             [hidden_dim],
                             h1_b_tm1, c1_b_tm1,
                             num_units,
                             input_mask=rev_inp_mask_t,
                             random_state=random_state,
                             cell_dropout=cell_dropout,
                             name=name + "reverse_rnn",
                             init=init)
        h1_b_t = s[0]
        c1_b_t = s[1]
        return h1_f_t, c1_f_t, h1_b_t, c1_b_t

    r = scan(step,
             [in_proj, input_mask, torch.flip(in_proj, (0,)), torch.flip(input_mask, (0,))],
             [h1_f_init, c1_f_init, h1_b_init, c1_b_init])
    return tf.concat([r[0], torch.flip(r[2], (0,))], axis=-1)

'''

class BiLSTMLayer(torch.nn.Module):
    def __init__(self,
                 list_of_input_dims,
                 num_units,
                 output_dim=None,
                 random_state=None,
                 name=None, init=None, scale="default",
                 forget_bias=1.,
                 strict=None):
        super(BiLSTMLayer, self).__init__()
        if name is None:
            name = _get_name()
        name = name + "_bidirlstm_layer"
        name_proj = name + "_proj"
        hidden_dim = 4 * num_units
        in_proj_obj = Linear(list_of_input_dims,
                             hidden_dim,
                             random_state=random_state,
                             name=name_proj,
                             init=init, strict=strict)

        fwd_cell_obj = LSTMCell([hidden_dim],
                                num_units,
                                random_state=random_state,
                                name=name + "forward_rnn",
                                init=init)

        rev_cell_obj = LSTMCell([hidden_dim],
                                 num_units,
                                 random_state=random_state,
                                 name=name + "reverse_rnn",
                                 init=init)

        self.in_proj_obj = in_proj_obj
        self.fwd_cell_obj = fwd_cell_obj
        self.rev_cell_obj = rev_cell_obj
        self.num_units = num_units

    def forward(self, list_of_inputs,
                previous_forward_hidden=None, previous_forward_cell=None,
                previous_reverse_hidden=None, previous_reverse_cell=None,
                input_mask=None,
                cell_dropout=None,
                strict=None):

        num_units = self.num_units
        if input_mask is None:
            raise ValueError("No input mask currently unsupported")

        in_proj = self.in_proj_obj(list_of_inputs)

        if previous_forward_hidden == None:
            h1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_f_init = previous_forward_hidden
        if previous_reverse_hidden == None:
            h1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            h1_b_init = previous_reverse_hidden
        if previous_forward_cell == None:
            c1_f_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_f_init = previous_forward_cell
        if previous_reverse_cell == None:
            c1_b_init = 0. * in_proj[0, :, :num_units].detach()
        else:
            c1_b_init = previous_reverse_cell

        def step(inp_t, inp_mask_t,
                 rev_inp_t, rev_inp_mask_t,
                 h1_f_tm1, c1_f_tm1, h1_b_tm1, c1_b_tm1):
            output, s = self.fwd_cell_obj([inp_t],
                                          h1_f_tm1, c1_f_tm1,
                                          input_mask=inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_f_t = s[0]
            c1_f_t = s[1]

            output, s = self.rev_cell_obj([rev_inp_t],
                                          h1_b_tm1, c1_b_tm1,
                                          input_mask=rev_inp_mask_t,
                                          cell_dropout=cell_dropout)
            h1_b_t = s[0]
            c1_b_t = s[1]
            return h1_f_t, c1_f_t, h1_b_t, c1_b_t

        r = scan(step,
                 [in_proj, input_mask, torch.flip(in_proj, (0,)), torch.flip(input_mask, (0,))],
                 [h1_f_init, c1_f_init, h1_b_init, c1_b_init])
        return torch.cat([r[0], torch.flip(r[2], (0,))], dim=-1)
