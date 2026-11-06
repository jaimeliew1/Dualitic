import functools

import numpy as np
from scipy import special, integrate, interpolate
from itertools import repeat

REGISTERED_DUAL_UFUNCS = {}
MONKEY_PATCHED = []


def DualVariables(variables):
    """
    Constructs dual variables from a list of primal variables.

    For each primal variable, creates a corresponding dual variable
    with the dual component set to 1 at the index matching the
    primal variable's position in the input list. All other dual
    components are set to 0.

    Args:
        variables (list): List of primal variables

    Returns:
        list: List of DualNumber objects
    """
    variables = list(variables)
    N = len(variables)

    out = []
    for i, variable in enumerate(variables):
        dual = np.zeros(N)
        dual[i] = 1
        out.append(DualNumber(variable, dual))
    return out


def undual(x):
    """
    'unduals' a number returning the real part if the number is a DualNumber, or
    just returns the value again if it is not.
    """
    if isinstance(x, DualNumber):
        return x.real[0]
    return x


class DualNumber(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, real, dual):
        self.real = np.atleast_1d(real)
        self.dual = np.atleast_2d(dual)

    @property
    def degree(self):
        """
        Returns the degree of the dual number. i.e. the number of dual variables.
        """
        return self.dual.shape[-1]

    @property
    def size(self):
        return self.real.size

    @property
    def shape(self):
        return self.real.shape

    @property
    def ndim(self):
        return self.real.ndim

    def __repr__(self):
        return f"DualNumber(degree={self.degree}, {self.real}, {self.dual})"

    def __iter__(self):
        return (DualNumber(r, d) for r, d in zip(self.real, self.dual))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return DualNumber(self.real[index], self.dual[(*index, slice(None))])
        else:
            return DualNumber(self.real[index], self.dual[(index, slice(None))])

    @staticmethod
    def _fit_to(target, value):
        # numpy >= 2.4 errors on assigning a shape-(1,) array to a scalar slot
        value = np.asarray(value)
        if value.size == np.size(target) and value.shape != np.shape(target):
            return value.reshape(np.shape(target))
        return value

    def __setitem__(self, index, newvalue):
        # if isinstance(index, tuple):
        #     raise NotImplementedError
        # else:
        #     if isinstance(newvalue, DualNumber):
        #         self.real[index] = newvalue.real
        #         self.dual[index] = newvalue.dual
        #     else:
        #         self.real[index] = newvalue
        #         self.dual[index] = 0.0
        if isinstance(newvalue, DualNumber):
            self.real[index] = self._fit_to(self.real[index], newvalue.real)
            self.dual[index] = newvalue.dual
        else:
            self.real[index] = self._fit_to(self.real[index], newvalue)
            if isinstance(index, tuple):
                self.dual[(*index, slice(None))] = 0.0
            else:
                self.dual[index] = 0.0

    def flatten(self, order="C"):
        real = self.real.flatten(order)
        dual = np.reshape(self.dual, (*real.shape, -1), order)

        return DualNumber(real, dual)

    def __add__(self, other):
        if isinstance(other, DualNumber):
            real = self.real + other.real
            dual = self.dual + other.dual
        elif isinstance(other, np.ndarray):
            real = self.real + other
            dual = np.zeros_like(other[..., None]) + self.dual

        else:
            real = self.real + other
            dual = self.dual.copy()  # a new array, like numpy; x + 0 must not alias x
        return DualNumber(real, dual)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            real = self.real - other.real
            dual = self.dual - other.dual
        elif isinstance(other, np.ndarray):
            real = self.real - other
            dual = np.zeros_like(other[..., None]) + self.dual
        else:
            real = self.real - other
            dual = self.dual.copy()  # a new array, like numpy; x - 0 must not alias x
        return DualNumber(real, dual)

    def __rsub__(self, other):
        real = other - self.real
        dual = -self.dual
        return DualNumber(real, dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            real = self.real * other.real
            dual = self.real[..., None] * other.dual + self.dual * other.real[..., None]
        elif isinstance(other, np.ndarray):
            real = self.real * other
            dual = other[..., None] * self.dual
        else:
            real = self.real * other
            dual = self.dual * other
        return DualNumber(real, dual)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            real = self.real / other.real
            dual = (
                self.dual * other.real[..., None] - self.real[..., None] * other.dual
            ) / (other.real[..., None] ** 2)
        elif isinstance(other, np.ndarray):
            # breakpoint()
            real = self.real / other
            dual = (self.dual * other[..., None]) / (other[..., None] ** 2)
        else:
            real = self.real / other
            dual = self.dual / other
        return DualNumber(real, dual)

    def __rtruediv__(self, other):
        real = other / self.real
        dual = -other * self.dual / (self.real[..., None] ** 2)
        return DualNumber(real, dual)

    def __floordiv__(self, other):
        if isinstance(other, DualNumber):
            real = self.real // other.real
            dual = self.dual - (self.real % other.real) * other.dual
        else:
            real = self.real // other
            dual = self.dual
        return DualNumber(real, dual)

    def __rfloordiv__(self, other):
        real = other // self.real
        dual = -(other % self.real) * self.dual
        return DualNumber(real, dual)

    def __mod__(self, other):
        if isinstance(other, DualNumber):
            real = self.real % other.real
            dual = self.dual
        else:
            real = self.real % other
            dual = self.dual
        return DualNumber(real, dual)

    def __rmod__(self, other):
        real = other % self.real
        dual = -other * self.dual / self.real
        return DualNumber(real, dual)

    def __pow__(self, exponent):
        if isinstance(exponent, DualNumber):
            # Formula: (x^y)' = (x^y) * (y' * log(x) + y * x' / x)
            real_exp = self.real**exponent.real
            log_term = exponent.dual * np.log(self.real)
            ratio_term = exponent.real * self.dual / self.real
            dual_exp = real_exp * (log_term + ratio_term)
        else:
            real_exp = self.real**exponent
            dual_exp = exponent * self.real[..., None] ** (exponent - 1) * self.dual
        return DualNumber(real_exp, dual_exp)

    def __neg__(self):
        return DualNumber(-self.real, -self.dual)

    def __pos__(self):
        raise NotImplementedError

    def __abs__(self):
        return DualNumber(abs(self.real), np.sign(self.real)[..., None] * self.dual)

    def __lt__(self, other):
        return self.real < other.real

    def __le__(self, other):
        return self.real <= other.real

    def __gt__(self, other):
        return self.real > other.real

    def __ge__(self, other):
        return self.real >= other.real

    def __eq__(self, other):
        return self.real == other.real

    def __ne__(self, other):
        return self.real != other.real

    # NOTE: `method` and `kwargs` are silently ignored, so ufunc methods and options are unsupported:
    # np.min(x) (np.minimum.reduce) returns x unreduced and np.divide(x, y, where=...) drops `where`.
    # Unregistered ufuncs do raise. Non-ufunc numpy functions that are not monkey patched below also
    # misbehave silently (e.g. np.nan_to_num, np.linalg.norm). See tests/bugs.md.
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in REGISTERED_DUAL_UFUNCS:
            return REGISTERED_DUAL_UFUNCS[ufunc](*inputs)
        else:
            raise NotImplementedError(f"{ufunc.__name__} not implemented")

    def squeeze(self, axis=None):
        if axis is None:
            real = self.real.squeeze()
            dual = self.dual.squeeze(axis=tuple(
                i for i, s in enumerate(self.real.shape) if s == 1
            ))
        else:
            real = self.real.squeeze(axis=axis)
            # Adjust negative axis to avoid touching the trailing degree axis
            dual_axis = axis if axis >= 0 else axis - 1
            dual = self.dual.squeeze(axis=dual_axis)
        return DualNumber(real, dual)

    # def reshape(self, *shape, order="C"):
    #     # Allow both reshape(a, b, c) and reshape((a, b, c)) calling conventions
    #     if len(shape) == 1 and hasattr(shape[0], '__iter__'):
    #         shape = tuple(shape[0])
    #     real = self.real.reshape(shape, order=order)
    #     dual = self.dual.reshape((*shape, -1), order=order)
    #     return DualNumber(real, dual)

    def reshape(self, *shape, order="C"):
        # Allow both reshape(a, b, c) and reshape((a, b, c)) calling conventions
        if len(shape) == 1 and hasattr(shape[0], '__iter__'):
            shape = tuple(shape[0])
        real = self.real.reshape(shape, order=order)
        # Compute degree explicitly rather than using -1, since the shape
        # passed in may already contain a -1
        dual = self.dual.reshape((*real.shape, self.degree), order=order)
        return DualNumber(real, dual)


def register_dual_ufunc(ufunc):
    def decorator(func):
        REGISTERED_DUAL_UFUNCS[ufunc] = func
        return func

    return decorator


def monkey_patch(to_override):
    MONKEY_PATCHED.append(to_override)

    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.to_override = to_override

        return wrapper

    return decorator


def _only_for_dual(original, name="a", position=0):
    """
    Calls the decorated function only if the argument at `position` (or passed as
    keyword `name`) is a DualNumber. Any other call goes to `original` with its
    arguments unchanged, so importing dualitic does not alter numpy/scipy behaviour.
    """

    def decorator(dual_func):
        @functools.wraps(original)
        def override(*args, **kwargs):
            arg = args[position] if len(args) > position else kwargs.get(name)
            if isinstance(arg, DualNumber):
                return dual_func(*args, **kwargs)
            return original(*args, **kwargs)

        return override

    return decorator


@register_dual_ufunc(np.absolute)
def _(self):
    return abs(self)


@register_dual_ufunc(np.add)
def _(self, other):
    return other + self


@register_dual_ufunc(np.subtract)
def _(self, other):
    return self + -other


@register_dual_ufunc(np.multiply)
def _(self, other):
    return other * self


@register_dual_ufunc(np.true_divide)
def _(self, other):
    if isinstance(self, np.ndarray) or isinstance(self, np.float64):
        real = self / other.real
        dual = -self[..., None] * other.dual / (other.real[..., None] ** 2)
        return DualNumber(real, dual)
    if isinstance(other, DualNumber):
        real = self.real / other.real
        dual = (self.dual * other.real - self.real * other.dual) / (other.real**2)
    else:
        real = self.real / other
        dual = self.dual / other
    return DualNumber(real, dual)


@register_dual_ufunc(np.cos)
def _(x):
    return DualNumber(np.cos(x.real), -x.dual * np.sin(x.real[..., None]))


@register_dual_ufunc(np.sin)
def _(x):
    return DualNumber(np.sin(x.real), x.dual * np.cos(x.real[..., None]))


@register_dual_ufunc(np.arctan2)
def _(x, y):
    if isinstance(x, DualNumber):
        xr, xd = x.real, x.dual
    else:
        xr, xd = x, np.zeros_like(x)[..., None]
    if isinstance(y, DualNumber):
        yr, yd = y.real, y.dual
    else:
        yr, yd = y, np.zeros_like(y)[..., None]
    real = np.arctan2(xr, yr)
    denom = xr[..., None] ** 2 + yr[..., None] ** 2
    # derivative is undefined at the origin and should probably be np.nan; 0 is kept so wake models can average over it
    safe_denom = np.where(denom == 0, 1.0, denom)
    dual = np.where(
        denom == 0, 0.0, (xd * yr[..., None] - xr[..., None] * yd) / safe_denom
    )
    return DualNumber(real, dual)


@register_dual_ufunc(np.arctan)
def _(x):
    real = np.arctan(x.real)
    dual = x.dual / (1 + x.real[..., None] ** 2)

    return DualNumber(real, dual)


@register_dual_ufunc(np.arccos)
def _(x):
    real = np.arccos(x.real)
    dual = -x.dual / np.sqrt(1 - x.real[..., None] ** 2)

    return DualNumber(real, dual)


@register_dual_ufunc(np.tanh)
def _(x):
    """
    Derivative of tanh(x) is 1 - tanh(x)^2
    """
    real = np.tanh(x.real)
    # We use [..., None] to align the primal dimensions with the dual vector
    dual = x.dual * (1 - real[..., None]**2)
    return DualNumber(real, dual)


@register_dual_ufunc(np.sign)
def _(x):
    """
    Derivative of sign(x) is 0 (almost everywhere)
    """
    real = np.sign(x.real)
    return DualNumber(real, np.zeros_like(x.dual))

@register_dual_ufunc(np.sqrt)
def _(x):
    real = np.sqrt(x.real)
    # breakpoint()
    return DualNumber(real, x.dual / (2 * real[..., None]))


@register_dual_ufunc(np.exp)
def _(x):
    real = np.exp(x.real)
    return DualNumber(real, x.dual * real[..., None])


@register_dual_ufunc(np.power)
def _(x1, x2):
    degree = x1.degree if isinstance(x1, DualNumber) else x2.degree
    if isinstance(x1, DualNumber):
        x1_real, x1_dual = x1.real, x1.dual
    else:
        x1_real, x1_dual = x1, np.zeros((*x1.shape, degree))

    if isinstance(x2, DualNumber):
        x2_real, x2_dual = x2.real, x2.dual
    else:
        x2_real, x2_dual = x2, np.zeros((*x2.shape, degree))

    real = x1_real**x2_real
    dual = real[..., None] * (
        x1_dual * x2_real[..., None] / x1_real[..., None]
        + x2_dual * np.log(x1_real[..., None])
    )
    return DualNumber(real, dual)


@register_dual_ufunc(np.log)
def _(x):
    return DualNumber(np.log(x.real), x.dual / x.real[..., None])


def _register_comparison(ufunc):
    @register_dual_ufunc(ufunc)
    def _(x1, x2, *args, **kwargs):
        if isinstance(x1, DualNumber):
            x1 = x1.real

        if isinstance(x2, DualNumber):
            x2 = x2.real

        return ufunc(x1, x2, *args, **kwargs)


for _ufunc in (
    np.greater_equal,
    np.greater,
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
):
    _register_comparison(_ufunc)


# @register_dual_ufunc(np.max)
# def _(x, axis=None, **kwargs):
#     return DualNumber(np.max(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))

# Monkey patch np.max instead
_max = np.max

@_only_for_dual(_max)
def max_override(x, axis=None, **kwargs):
    if axis is None:
        # Flatten spatial dims, find argmax in the flat real array,
        # then grab the corresponding dual row
        flat_real = x.real.flatten()
        flat_dual = x.dual.reshape(-1, x.degree)
        idx = np.argmax(flat_real)
        return DualNumber(flat_real[idx:idx+1], flat_dual[idx:idx+1])

    # Reduce along a specific axis
    idx = np.argmax(x.real, axis=axis)          # shape: (...) with `axis` removed
    real = _max(x.real, axis=axis, **kwargs)    # same shape as idx

    # Expand idx to index into the spatial axes of dual, then append
    # a slice for the trailing degree axis
    idx_expanded = np.expand_dims(idx, axis=axis)   # reinsert reduced axis
    dual_axis = axis if axis >= 0 else axis - 1     # account for degree axis
    idx_dual = np.expand_dims(idx_expanded, axis=-1)  # (..., 1, ..., 1) for degree
    idx_dual = np.broadcast_to(idx_dual, (*idx_expanded.shape, x.degree))

    dual = np.take_along_axis(x.dual, idx_dual, axis=dual_axis)
    dual = dual.squeeze(axis=dual_axis)

    return DualNumber(real, dual)

np.max = max_override


@register_dual_ufunc(np.min)
def _(x, axis=None, **kwargs):
    return DualNumber(np.min(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))


@register_dual_ufunc(np.maximum)
def _(a, b=None, *args, **kwargs):
    if b is None:
        return a

    if isinstance(a, DualNumber):
        a_real, a_dual = a.real, a.dual
    else:
        b_real_ref = b.real if isinstance(b, DualNumber) else b
        a = np.asarray(a)
        degree = b.degree if isinstance(b, DualNumber) else a.degree
        a_real = a
        a_dual = np.zeros((*a.shape, degree))

    if isinstance(b, DualNumber):
        b_real, b_dual = b.real, b.dual
    else:
        b = np.asarray(b)
        b_real = b
        b_dual = np.zeros((*b.shape, a_dual.shape[-1]))

    # NaN propagates in either argument, as in numpy. A tie takes the derivative of b, so
    # np.clip(x, lo, hi) has zero derivative exactly at a constant bound.
    out_real = np.maximum(a_real, b_real)
    dual_indices = np.repeat((a_real > b_real)[..., np.newaxis], a_dual.shape[-1], axis=-1)
    out_dual = np.where(dual_indices, a_dual, b_dual)
    out_dual = np.where(np.isnan(out_real)[..., np.newaxis], np.nan, out_dual)

    return DualNumber(out_real, out_dual)


# @register_dual_ufunc(np.minimum)
# def _(a, b, *args, **kwargs):
#     assert isinstance(a, DualNumber)
#     assert isinstance(b, float) or isinstance(b, int)
#     indices = np.where(a.real > b)

#     out = a
#     out.real[indices] = b
#     out.dual[indices] = 0
#     return out

@register_dual_ufunc(np.minimum)
def _(a, b=None, *args, **kwargs):
    if b is None:
        return a

    if isinstance(a, DualNumber):
        a_real, a_dual = a.real, a.dual
    else:
        b_real_ref = b.real if isinstance(b, DualNumber) else b
        a = np.asarray(a)
        degree = b.degree if isinstance(b, DualNumber) else a.degree
        a_real = a
        a_dual = np.zeros((*a.shape, degree))

    if isinstance(b, DualNumber):
        b_real, b_dual = b.real, b.dual
    else:
        b = np.asarray(b)
        b_real = b
        b_dual = np.zeros((*b.shape, a_dual.shape[-1]))

    # NaN and tie handling as in np.maximum above
    out_real = np.minimum(a_real, b_real)
    dual_indices = np.repeat((a_real < b_real)[..., np.newaxis], a_dual.shape[-1], axis=-1)
    out_dual = np.where(dual_indices, a_dual, b_dual)
    out_dual = np.where(np.isnan(out_real)[..., np.newaxis], np.nan, out_dual)

    return DualNumber(out_real, out_dual)


@register_dual_ufunc(np.deg2rad)
def _(x, axis=None, **kwargs):
    return DualNumber(np.deg2rad(x.real), np.deg2rad(x.dual))


@register_dual_ufunc(np.rad2deg)
def _(x, axis=None, **kwargs):
    return DualNumber(np.rad2deg(x.real), np.rad2deg(x.dual))


@register_dual_ufunc(np.isnan)
def _(x, *args, **kwargs):
    return np.isnan(x.real, *args, **kwargs)


@register_dual_ufunc(np.isinf)
def _(x, *args, **kwargs):
    return np.isinf(x.real, *args, **kwargs)


@register_dual_ufunc(np.isfinite)
def _(x, *args, **kwargs):
    return np.isfinite(x.real, *args, **kwargs)


@register_dual_ufunc(np.nanmax)
def _(x, *args, **kwargs):
    return np.nanmax(x.real, *args, **kwargs)


@register_dual_ufunc(special.erf)
def _(x):
    real = special.erf(x.real)
    return DualNumber(
        real, (2 * x.dual / np.sqrt(np.pi)) * np.exp(-(x.real[..., None] ** 2))
    )


### Monkey patching

# Monkey patch np.zeros_like / np.ones_like / np.empty_like
# For dual input these return a DualNumber with zero derivatives (like numpy, the "_like" functions
# preserve the input type), so that code which preallocates and then fills slices keeps its derivatives.
def _like_override(original):
    @_only_for_dual(original)
    def override(a, *args, **kwargs):
        real = original(a.real, *args, **kwargs)
        return DualNumber(real, np.zeros((*real.shape, a.degree)))

    return override


_zeros_like = np.zeros_like
zeros_like_override = _like_override(_zeros_like)
np.zeros_like = zeros_like_override

_ones_like = np.ones_like
ones_like_override = _like_override(_ones_like)
np.ones_like = ones_like_override

_empty_like = np.empty_like
empty_like_override = _like_override(_empty_like)
np.empty_like = empty_like_override

# Monkey patch np.argsort
# _argsort = np.argsort


# def argsort_override(a, axis=-1, kind=None, order=None):
#     if not isinstance(a, DualNumber):
#         return _argsort(a, axis=axis, kind=kind, order=order)
#     elif isinstance(a, DualNumber):
#         return _argsort(a.real, axis=axis, kind=kind, order=order)


# np.argsort = argsort_override
# Capture the original only as a fallback for non-array types
_argsort = np.argsort

@_only_for_dual(_argsort)
def argsort_override(a, *args, **kwargs):
    return _argsort(a.real, *args, **kwargs)

np.argsort = argsort_override


_where = np.where

def where_override(condition, *args):
    # Mode 1: np.where(condition) -> returns indices
    if len(args) == 0:
        return _where(condition)
    
    if len(args) != 2:
        raise TypeError(f"where() takes 1 or 3 arguments ({len(args) + 1} given)")

    x, y = args
    is_x_dual = isinstance(x, DualNumber)
    is_y_dual = isinstance(y, DualNumber)
    
    if not is_x_dual and not is_y_dual:
        return _where(condition, x, y)

    # 1. Normalize the condition to an array
    # This turns True -> array(True), which supports slicing/broadcasting
    cond_arr = np.asanyarray(condition)

    # 2. Extract real parts
    x_real = x.real if is_x_dual else x
    y_real = y.real if is_y_dual else y
    res_real = _where(cond_arr, x_real, y_real)

    # 3. Determine degree and prepare dual parts
    # We assume x and y have the same degree if both are DualNumbers
    degree = x.degree if is_x_dual else y.degree
    
    x_dual = x.dual if is_x_dual else np.zeros((*np.shape(x_real), degree))
    y_dual = y.dual if is_y_dual else np.zeros((*np.shape(y_real), degree))
    
    # 4. Broadcast the mask to the dual dimension
    # cond_arr[..., None] adds a trailing dimension of size 1
    # This allows the mask to be applied across all 'degree' derivatives
    res_dual = _where(cond_arr[..., None], x_dual, y_dual)

    return DualNumber(res_real, res_dual)

np.where = where_override


# Monkey patch np.sum
_sum = np.sum


# def sum_override(x, axis=None, **kwargs):
#     if not isinstance(x, DualNumber):
#         return _sum(x, axis=axis, **kwargs)
#     else:
#         if axis is None:
#             axis = tuple(range(x.real.ndim))
#         return DualNumber(
#             _sum(x.real, axis=axis, **kwargs), _sum(x.dual, axis=axis, **kwargs)
#         )
@_only_for_dual(_sum)
def sum_override(x, axis=None, **kwargs):
    real = _sum(x.real, axis=axis, **kwargs)
    
    if axis is None:
        dual_axis = tuple(range(x.real.ndim))
    elif isinstance(axis, tuple):
        dual_axis = tuple(ax if ax >= 0 else ax - 1 for ax in axis)
    else:
        dual_axis = axis if axis >= 0 else axis - 1
    
    dual = _sum(x.dual, axis=dual_axis, **kwargs)
    return DualNumber(real, dual)

np.sum = sum_override

# Monkey patch np.mean
_mean = np.mean


# def mean_override(x, axis=None, **kwargs):
#     if not isinstance(x, DualNumber):
#         return _mean(x, axis=axis, **kwargs)
#     else:
#         if axis is None:
#             axis = tuple(range(x.real.ndim))
#         return DualNumber(
#             _mean(x.real, axis=axis, **kwargs), _mean(x.dual, axis=axis, **kwargs)
#         )
@_only_for_dual(_mean)
def mean_override(x, axis=None, **kwargs):
    real = _mean(x.real, axis=axis, **kwargs)
    
    if axis is None:
        dual_axis = tuple(range(x.real.ndim))
    elif isinstance(axis, tuple):
        dual_axis = tuple(ax if ax >= 0 else ax - 1 for ax in axis)
    else:
        dual_axis = axis if axis >= 0 else axis - 1
    
    dual = _mean(x.dual, axis=dual_axis, **kwargs)
    return DualNumber(real, dual)

np.mean = mean_override

# Monkey patch np.meshgrid
_meshgrid = np.meshgrid


# def meshgrid_override(*xi, indexing="xy", **kwargs):
#     if len(xi) > 2:
#         raise NotImplementedError
#     if indexing != "xy":
#         raise NotImplementedError

#     x, y = xi
#     if not any(isinstance(arg, DualNumber) for arg in [x, y]):
#         return _meshgrid(*xi, indexing="xy", **kwargs)

#     # Generate real part
#     x_real = x.real if isinstance(x, DualNumber) else x
#     y_real = y.real if isinstance(y, DualNumber) else y

#     x_len = len(x_real)
#     y_len = len(y_real)

#     xs_real, ys_real = _meshgrid(x_real, y_real)

#     # Generate dual part
#     if isinstance(x, DualNumber):
#         x_dual_lg = x.dual[None, ...]
#         xs_dual = np.repeat(x_dual_lg, y_len, axis=0)
#         x_return = DualNumber(xs_real, xs_dual)
#     else:
#         x_return = xs_real

#     if isinstance(y, DualNumber):
#         y_dual_lg = y.dual[..., None, :]
#         concat_tuple = tuple(repeat(y_dual_lg, x_len))
#         ys_dual = np.concatenate(concat_tuple, axis=1)
#         y_return = DualNumber(ys_real, ys_dual)
#     else:
#         y_return = ys_real

#     return x_return, y_return
def meshgrid_override(*xi, **kwargs):
    if not any(isinstance(x, DualNumber) for x in xi):
        return _meshgrid(*xi, **kwargs)

    indexing = kwargs.pop("indexing", "xy")

    # 1. Extract real parts and identify DualNumbers
    is_dual = [isinstance(x, DualNumber) for x in xi]
    real_inputs = [x.real if isinstance(x, DualNumber) else x for x in xi]
    
    # 2. Get the real-valued meshgrids from NumPy
    # This handles the complex logic of 'xy' vs 'ij' for us
    real_grids = _meshgrid(*real_inputs, indexing=indexing, **kwargs)
    
    # The shape of the resulting grid (e.g., (Ny, Nx) for 'xy' or (Nx, Ny) for 'ij')
    grid_shape = real_grids[0].shape
    
    # Find the 'degree' (number of dual variables) from any dual input
    degree = next((x.degree for x in xi if isinstance(x, DualNumber)), None)

    results = []
    for i, (grid_real, original_input) in enumerate(zip(real_grids, xi)):
        if not isinstance(original_input, DualNumber):
            results.append(grid_real)
            continue
            
        # 3. Generalized Broadcasting for the Dual Part
        # We need to reshape the 1D dual part (Length, Degree) 
        # so that 'Length' aligns with the correct axis in the grid.
        
        # Determine which axis the current input 'i' corresponds to
        if indexing == "xy" and len(xi) >= 2:
            if i == 0: target_axis = 1
            elif i == 1: target_axis = 0
            else: target_axis = i
        else: # "ij" indexing or 1D case
            target_axis = i

        # Create a reshape dimensions list: [1, 1, ..., Length_i, ..., 1, Degree]
        # There are len(xi) spatial dimensions + 1 degree dimension
        reshape_dims = [1] * (len(xi) + 1)
        reshape_dims[target_axis] = len(original_input.real)
        reshape_dims[-1] = degree
        
        # Reshape and broadcast the dual component to the full grid size
        dual_reshaped = original_input.dual.reshape(reshape_dims)
        grid_dual = np.broadcast_to(dual_reshaped, (*grid_shape, degree))
        
        results.append(DualNumber(grid_real, grid_dual))

    return tuple(results)

np.meshgrid = meshgrid_override

np.meshgrid = meshgrid_override


# Monkey patch np.interp
_interp = np.interp


def interp_override(x, xp, fp, *args, **kwargs):
    if not any(isinstance(arg, DualNumber) for arg in [x, xp, fp]):
        return _interp(x, xp, fp, *args, **kwargs)

    if isinstance(x, DualNumber):
        x_real = x.real[0]
    else:
        x_real = x

    if isinstance(xp, DualNumber):
        idx = np.searchsorted(xp.real, x_real) - 1
    else:
        idx = np.searchsorted(xp, x_real) - 1

    x_l, x_r = xp[idx], xp[idx + 1]
    y_l, y_r = fp[idx], fp[idx + 1]

    return ((y_r - y_l) / (x_r - x_l)) * (x - x_l) + y_l


np.interp = interp_override

# Monkey patch np.clip
_clip = np.clip


@_only_for_dual(_clip)
def clip_override(x, a_min=None, a_max=None):
    return np.minimum(np.maximum(x, a_min), a_max)


np.clip = clip_override

# Monkey patch np.expand_dims
_expand_dims = np.expand_dims


def expand_dims_override(a, axis):
    if isinstance(a, DualNumber):
        real = _expand_dims(a.real, axis)
        if axis < 0:
            axis -= 1
        dual = _expand_dims(a.dual, axis)
        return DualNumber(real, dual)
    else:
        return _expand_dims(a, axis)


np.expand_dims = expand_dims_override

# Monkey patch np.squeeze
_squeeze = np.squeeze


# def squeeze_override(x, axis=None, **kwargs):
#     if isinstance(x, DualNumber):
#         return DualNumber(np.squeeze(x.real, -1), np.squeeze(x.dual, -2))
#     else:
#         return _squeeze(x, axis=axis, **kwargs)
# Fix np.squeeze
@_only_for_dual(_squeeze)
def squeeze_override(x, axis=None):
    return x.squeeze(axis=axis)

np.squeeze = squeeze_override


# Monkey patch np.reshape + fix
_reshape = np.reshape

@_only_for_dual(_reshape)
def reshape_override(a, shape=None, order="C", *, newshape=None):
    return a.reshape(newshape if shape is None else shape, order=order)

np.reshape = reshape_override


# Monkey patch np.diff
_diff = np.diff

def diff_override(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
    if not isinstance(a, DualNumber):
        return _diff(a, n=n, axis=axis, prepend=prepend, append=append)
    
    real = _diff(a.real, n=n, axis=axis, prepend=prepend, append=append)
    # dual has an extra trailing degree axis, so shift the spatial axis by -1
    # to avoid touching it
    dual_axis = axis if axis >= 0 else axis - 1
    dual = _diff(a.dual, n=n, axis=dual_axis, prepend=prepend, append=append)
    return DualNumber(real, dual)

np.diff = diff_override


# Monkey patch np.gradient
_gradient = np.gradient

def gradient_override(f, *varargs, axis=None, edge_order=1, **kwargs):
    if not isinstance(f, DualNumber):
        return _gradient(f, *varargs, axis=axis, edge_order=edge_order, **kwargs)

    real = _gradient(f.real, *varargs, axis=axis, edge_order=edge_order, **kwargs)

    # gradient returns a list (tuple on numpy >= 2) when axis=None or axis is a
    # tuple, and a single array when axis is an integer
    if isinstance(real, (list, tuple)):
        # Compute dual for each returned gradient component
        dual_results = []
        for i, ax in enumerate(range(f.real.ndim) if axis is None else axis):
            dual_ax = ax if ax >= 0 else ax - 1
            dual = _gradient(f.dual, *varargs, axis=dual_ax, 
                             edge_order=edge_order, **kwargs)
            # gradient returns a list here too — pick the component for this axis
            if isinstance(dual, (list, tuple)):
                dual_results.append(DualNumber(real[i], dual[dual_ax]))
            else:
                dual_results.append(DualNumber(real[i], dual))
        return type(real)(dual_results)
    else:
        dual_axis = axis if axis >= 0 else axis - 1
        dual = _gradient(f.dual, *varargs, axis=dual_axis, 
                         edge_order=edge_order, **kwargs)
        return DualNumber(real, dual)

np.gradient = gradient_override


# Monkey patch np.concatenate
_concatenate = np.concatenate

def concatenate_override(arrays, axis=0, *args, **kwargs):
    if not any(isinstance(a, DualNumber) for a in arrays):
        return _concatenate(arrays, axis, *args, **kwargs)

    # Promote any plain arrays to DualNumbers with zero duals
    degree = next(a.degree for a in arrays if isinstance(a, DualNumber))
    promoted = []
    for a in arrays:
        if isinstance(a, DualNumber):
            promoted.append(a)
        else:
            a = np.asarray(a)
            promoted.append(DualNumber(a, np.zeros((*a.shape, degree))))

    real = _concatenate([a.real for a in promoted], axis, *args, **kwargs)
    # dual has extra trailing degree axis, so shift spatial axis accordingly
    dual_axis = axis if axis >= 0 else axis - 1
    dual = _concatenate([a.dual for a in promoted], dual_axis, *args, **kwargs)
    return DualNumber(real, dual)

np.concatenate = concatenate_override


# Monkey patch np.broadcast_arrays
_broadcast_arrays = np.broadcast_arrays

def broadcast_arrays_override(*args, **kwargs):
    if not any(isinstance(a, DualNumber) for a in args):
        return _broadcast_arrays(*args, **kwargs)

    # Determine degree from any dual input
    degree = next(a.degree for a in args if isinstance(a, DualNumber))

    # Promote plain arrays to DualNumbers with zero duals
    promoted = []
    for a in args:
        if isinstance(a, DualNumber):
            promoted.append(a)
        else:
            a = np.asarray(a)
            promoted.append(DualNumber(a, np.zeros((*a.shape, degree))))

    # Broadcast real parts to get target shape
    real_arrays = _broadcast_arrays(*[a.real for a in promoted], **kwargs)
    target_shape = real_arrays[0].shape

    # Broadcast dual parts along spatial axes only, leaving degree axis untouched
    results = []
    for real, orig in zip(real_arrays, promoted):
        dual = np.broadcast_to(orig.dual, (*target_shape, degree))
        results.append(DualNumber(real, dual))

    return results

np.broadcast_arrays = broadcast_arrays_override


# Monkey patch np.pad
_pad = np.pad

def pad_override(array, pad_width, mode='constant', **kwargs):
    if not isinstance(array, DualNumber):
        return _pad(array, pad_width, mode=mode, **kwargs)

    real = _pad(array.real, pad_width, mode=mode, **kwargs)
    
    # dual has an extra trailing degree axis, so append (0, 0) padding
    # for that axis to leave it untouched
    if isinstance(pad_width, int):
        dual_pad_width = [(pad_width, pad_width)] * array.real.ndim + [(0, 0)]
    elif isinstance(pad_width, tuple) and isinstance(pad_width[0], int):
        # (before, after) applied to all axes
        dual_pad_width = [pad_width] * array.real.ndim + [(0, 0)]
    else:
        # sequence of (before, after) per axis — just append (0, 0) for degree axis
        dual_pad_width = list(pad_width) + [(0, 0)]

    dual = _pad(array.dual, dual_pad_width, mode=mode, **kwargs)
    return DualNumber(real, dual)

np.pad = pad_override


# Monkey patch np.hstack
_hstack = np.hstack

def hstack_override(tup, **kwargs):
    if not any(isinstance(a, DualNumber) for a in tup):
        return _hstack(tup, **kwargs)

    degree = next(a.degree for a in tup if isinstance(a, DualNumber))
    promoted = []
    for a in tup:
        if isinstance(a, DualNumber):
            promoted.append(a)
        else:
            a = np.asarray(a)
            promoted.append(DualNumber(a, np.zeros((*a.shape, degree))))

    # hstack concatenates along axis=0 for 1D, axis=1 for 2D+
    # mimic this by delegating to concatenate_override
    if all(a.real.ndim == 1 for a in promoted):
        return concatenate_override(promoted, axis=0)
    else:
        return concatenate_override(promoted, axis=1)

np.hstack = hstack_override


# Monkey patch np.vstack
_vstack = np.vstack

def vstack_override(tup, **kwargs):
    if not any(isinstance(a, DualNumber) for a in tup):
        return _vstack(tup, **kwargs)

    degree = next(a.degree for a in tup if isinstance(a, DualNumber))
    promoted = []
    for a in tup:
        if isinstance(a, DualNumber):
            promoted.append(a)
        else:
            a = np.asarray(a)
            promoted.append(DualNumber(a, np.zeros((*a.shape, degree))))

    # vstack always concatenates along axis=0, but first promotes
    # 1D arrays to row vectors (1, N) — same as numpy's behaviour
    reshaped = []
    for a in promoted:
        if a.real.ndim == 1:
            reshaped.append(a.reshape(1, -1))
        else:
            reshaped.append(a)

    return concatenate_override(reshaped, axis=0)

np.vstack = vstack_override


# Monkey patch np.stack
_stack = np.stack

def stack_override(arrays, axis=0, *args, **kwargs):
    if not any(isinstance(a, DualNumber) for a in arrays):
        return _stack(arrays, axis, *args, **kwargs)

    degree = next(a.degree for a in arrays if isinstance(a, DualNumber))
    promoted = []
    for a in arrays:
        if isinstance(a, DualNumber):
            promoted.append(a)
        else:
            a = np.asarray(a)
            promoted.append(DualNumber(a, np.zeros((*a.shape, degree))))

    real = _stack([a.real for a in promoted], axis, *args, **kwargs)
    # Adjust axis for the trailing degree dimension
    dual_axis = axis if axis >= 0 else axis - 1
    dual = _stack([a.dual for a in promoted], dual_axis, *args, **kwargs)
    return DualNumber(real, dual)

np.stack = stack_override


# Monkey patch np.atleast_1d
_atleast_1d = np.atleast_1d

def atleast_1d_override(*arys):
    if not any(isinstance(a, DualNumber) for a in arys):
        return _atleast_1d(*arys)
    if len(arys) == 1:
        a = arys[0]
        if isinstance(a, DualNumber):
            return DualNumber(np.atleast_1d(a.real), np.atleast_2d(a.dual))
    return tuple(atleast_1d_override(a) for a in arys)

np.atleast_1d = atleast_1d_override


# Monkey patch np.atleast_2d
_atleast_2d = np.atleast_2d

def atleast_2d_override(*arys):
    if not any(isinstance(a, DualNumber) for a in arys):
        return _atleast_2d(*arys)
    if len(arys) == 1:
        a = arys[0]
        if isinstance(a, DualNumber):
            real = np.atleast_2d(a.real)
            dual = np.reshape(a.dual, (*real.shape, -1))
            return DualNumber(real, dual)
    return tuple(atleast_2d_override(a) for a in arys)

np.atleast_2d = atleast_2d_override


# Monkey patch np.atleast_3d
_atleast_3d = np.atleast_3d

def atleast_3d_override(*arys):
    if not any(isinstance(a, DualNumber) for a in arys):
        return _atleast_3d(*arys)
    if len(arys) == 1:
        a = arys[0]
        if isinstance(a, DualNumber):
            real = np.atleast_3d(a.real)
            dual = np.reshape(a.dual, (*real.shape, -1))
            return DualNumber(real, dual)
    return tuple(atleast_3d_override(a) for a in arys)

np.atleast_3d = atleast_3d_override

# Monkey patch trapezoidal integration: np.trapezoid (numpy >= 2), np.trapz
# (numpy < 2) and scipy.integrate.trapezoid.
def _make_trapz_override(base):
    @_only_for_dual(base, name="y")
    def trapz_override(y, x=None, dx=1.0, axis=-1):
        real = base(y.real, x=x, dx=dx, axis=axis)
        if axis < 0:
            axis -= 1
        x_dual = None if x is None else x[..., None]
        dual = base(y.dual, x=x_dual, dx=dx, axis=axis)
        return DualNumber(real, dual)

    return trapz_override


for _name in ("trapezoid", "trapz"):
    if hasattr(np, _name):
        setattr(np, _name, _make_trapz_override(getattr(np, _name)))

integrate.trapezoid = _make_trapz_override(integrate.trapezoid)


# Monkey patch cumulative_trapezoid
_cumulative_trapezoid = integrate.cumulative_trapezoid


@_only_for_dual(_cumulative_trapezoid, name="y")
def cumulative_trapezoid_override(y, x=None, dx=1.0, axis=-1, initial=None):
    real = _cumulative_trapezoid(y.real, x=x, dx=dx, axis=axis, initial=initial)
    if axis < 0:
        axis -= 1
    # a 1D x already follows `axis`; an N-D x needs the trailing degree axis
    x_dual = x if x is None or np.ndim(x) == 1 else np.asarray(x)[..., None]
    dual = _cumulative_trapezoid(
        y.dual, x=x_dual, dx=dx, axis=axis, initial=None if initial is None else 0
    )
    return DualNumber(real, dual)


integrate.cumulative_trapezoid = cumulative_trapezoid_override
integrate.cumtrapz = cumulative_trapezoid_override


# Monkey patch scipy.interpolate.interp1d

_interp1d = interpolate.interp1d


@_only_for_dual(_interp1d, name="y", position=1)
def interp1d_override(x, y, kind="linear", axis=-1, **kwargs):
    interp_real = _interp1d(x, y.real, kind, axis, **kwargs)
    if axis < 0:
        axis -= 1
    interp_dual = _interp1d(x, y.dual, kind, axis, **kwargs)
    return lambda x: DualNumber(interp_real(x), interp_dual(x))


interpolate.interp1d = interp1d_override

# Monkey patch scipy.interpolate.RectBivariateSpline

_RectBivariateSpline = interpolate.RectBivariateSpline


class RectBivariateSpline_override(_RectBivariateSpline):
    """
    Dual-number compatible RectBivariateSpline. Currently interpolates only real
    values onto a dual number-based interpolation grid.

    """

    def __call__(self, x, y, dx=0, dy=0, grid=True):
        if all(not isinstance(_x, DualNumber) for _x in (x, y)):
            return super().__call__(x, y, dx, dy, grid)
        else:
            primal = self(x.real, y.real, dx=dx, dy=dy, grid=grid)
            x_dual = x.dual if isinstance(x, DualNumber) else 0.0
            y_dual = y.dual if isinstance(y, DualNumber) else 0.0
            dual = (
                x_dual * self(x.real, y.real, dx=dx + 1, dy=dy, grid=grid)[..., None]
                + y_dual * self(x.real, y.real, dx=dx, dy=dy + 1, grid=grid)[..., None]
            )
            return DualNumber(primal, dual)


interpolate.RectBivariateSpline = RectBivariateSpline_override


def _query_point_slope(evaluate, grid, xi, i, extrapolate):
    """
    Central finite difference of an interpolant along dimension `i` of the query
    points `xi` (..., ndim). The stencil is kept inside the grid and becomes
    one-sided at a bound. Outside the grid the interpolant returns a constant
    fill_value, so the slope is 0, unless it is extrapolating.
    """
    eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.abs(xi[..., i]))
    xi_fwd = xi.copy()
    xi_bwd = xi.copy()
    xi_fwd[..., i] += eps
    xi_bwd[..., i] -= eps

    if extrapolate:
        return (evaluate(xi_fwd) - evaluate(xi_bwd)) / (2 * eps)

    inside = np.ones(xi.shape[:-1], dtype=bool)
    for dim, pts in enumerate(grid):
        lo, hi = min(pts[0], pts[-1]), max(pts[0], pts[-1])
        inside &= (xi[..., dim] >= lo) & (xi[..., dim] <= hi)
        xi_fwd[..., dim] = np.clip(xi_fwd[..., dim], lo, hi)
        xi_bwd[..., dim] = np.clip(xi_bwd[..., dim], lo, hi)

    # the denominator is the clamped width, not 2 * eps
    width = xi_fwd[..., i] - xi_bwd[..., i]
    slope = (evaluate(xi_fwd) - evaluate(xi_bwd)) / np.where(width > 0, width, 1.0)
    return np.where(inside, slope, 0.0)


# Monkey patch scipy.interpolate.RegularGridInterpolator

_RegularGridInterpolator = interpolate.RegularGridInterpolator


class RegularGridInterpolator_override(_RegularGridInterpolator):
    """
    Dual-number compatible RegularGridInterpolator. Supports interpolation of
    real-valued grids at dual-number query points, propagating derivatives via
    finite differences of the interpolant along each input dimension.
    """

    def __call__(self, xi, method=None, **kwargs):
        # --- Normalize input to a single DualNumber ---
        # Handle list/tuple of DualNumbers (output of DualVariables())
        # Each element may be a scalar DualNumber or a batched DualNumber
        if isinstance(xi, (list, tuple)) and any(isinstance(x, DualNumber) for x in xi):
            degree = next(x.degree for x in xi if isinstance(x, DualNumber))
            reals = np.stack([
                x.real if isinstance(x, DualNumber) else np.atleast_1d(x)
                for x in xi
            ], axis=-1)  # (..., ndim)
            duals = np.stack([
                x.dual if isinstance(x, DualNumber) else np.zeros((*np.atleast_1d(x).shape, degree))
                for x in xi
            ], axis=-2)  # (..., ndim, degree)
            xi = DualNumber(reals, duals)

        if not isinstance(xi, DualNumber):
            return super().__call__(xi, method=method, **kwargs)

        # Ensure xi_real is a plain float array of shape (..., ndim)
        xi_real = np.asarray(xi.real, dtype=float)
        if xi_real.ndim == 1:
            xi_real = xi_real[np.newaxis, :]  # (ndim,) -> (1, ndim)

        # xi.dual shape: (..., ndim, degree)
        xi_dual = xi.dual
        if xi_dual.ndim == 2:
            xi_dual = xi_dual[np.newaxis, :]  # (ndim, degree) -> (1, ndim, degree)

        ndim = xi_real.shape[-1]
        degree = xi.degree

        evaluate = functools.partial(super().__call__, method=method, **kwargs)
        primal = evaluate(xi_real)  # (...,)

        # --- Dual propagation ---
        dual = np.zeros((*primal.shape, degree))
        extrapolate = not self.bounds_error and self.fill_value is None

        for i in range(ndim):
            dfdxi = _query_point_slope(evaluate, self.grid, xi_real, i, extrapolate)  # (...,)

            # xi_dual[..., i, :] shape: (..., degree)
            dual += dfdxi[..., None] * xi_dual[..., i, :]

        return DualNumber(primal, dual)

interpolate.RegularGridInterpolator = RegularGridInterpolator_override


# Monkey patch scipy.interpolate.make_interp_spline
_make_interp_spline = interpolate.make_interp_spline

@_only_for_dual(_make_interp_spline, name="y", position=1)
def make_interp_spline_override(x, y, *args, **kwargs):
    # y is dual — spline interpolation is linear in y, so interpolate
    # real and each dual slice independently (exact, no finite differences)
    spline_real = _make_interp_spline(x, y.real, *args, **kwargs)
    spline_dual = [_make_interp_spline(x, y.dual[..., k_], *args, **kwargs)
                   for k_ in range(y.degree)]

    def call(t):
        t_real = t.real if isinstance(t, DualNumber) else t
        real = spline_real(t_real)
        dual = np.stack([s(t_real) for s in spline_dual], axis=-1)
        return DualNumber(real, dual)

    return call

interpolate.make_interp_spline = make_interp_spline_override


# Monkey patch scipy.interpolate.interpn
_interpn = interpolate.interpn
def interpn_override(points, values, xi, method="linear", bounds_error=True, fill_value=np.nan, **kwargs):
    # Extract real values immediately — must happen before any _interpn call
    values_dual = None
    if isinstance(values, DualNumber):
        values_dual = values.dual  # (nx, ny, nz, degree)
        values = values.real

    # Handle tuple/list of coordinate arrays with DualNumbers
    if isinstance(xi, (list, tuple)) and any(isinstance(x, DualNumber) for x in xi):
        degree = next(x.degree for x in xi if isinstance(x, DualNumber))
        reals = np.stack([
            x.real.flatten() if isinstance(x, DualNumber) else np.atleast_1d(x).flatten()
            for x in xi
        ], axis=-1)
        duals = np.stack([
            x.dual if isinstance(x, DualNumber) else np.zeros((*np.atleast_1d(x).flatten().shape, degree))
            for x in xi
        ], axis=-2)
        xi = DualNumber(reals, duals)

    if not isinstance(xi, DualNumber) and values_dual is None:
        return _interpn(points, values, xi, method=method,
                        bounds_error=bounds_error, fill_value=fill_value, **kwargs)

    xi_real = np.asarray(xi.real, dtype=float) if isinstance(xi, DualNumber) else xi
    if isinstance(xi_real, np.ndarray) and xi_real.ndim == 1:
        xi_real = xi_real[np.newaxis, :]

    primal = _interpn(points, values, xi_real, method=method,
                      bounds_error=bounds_error, fill_value=fill_value, **kwargs)

    # Determine degree from whichever is dual
    degree = (xi.degree if isinstance(xi, DualNumber) 
              else values_dual.shape[-1])
    dual = np.zeros((*primal.shape, degree))

    # --- Sensitivity w.r.t. xi (query points) ---
    if isinstance(xi, DualNumber):
        xi_dual = xi.dual
        if xi_dual.ndim == 2:
            xi_dual = xi_dual[np.newaxis, :]
        ndim = xi_real.shape[-1]

        evaluate = functools.partial(_interpn, points, values, method=method,
                                     bounds_error=bounds_error, fill_value=fill_value, **kwargs)
        extrapolate = not bounds_error and fill_value is None

        for i in range(ndim):
            dfdxi = _query_point_slope(evaluate, points, xi_real, i, extrapolate)

            dual += dfdxi[..., None] * xi_dual[..., i, :]

    # --- Sensitivity w.r.t. values (the grid data) ---
    # interpn is linear in values, so d/d(values_k) [interpn(points, values, xi)]
    # is just interpn(points, e_k, xi) where e_k is the k-th basis vector.
    # Equivalently: interpolate each dual slice of values independently.
    if values_dual is not None:
        for k in range(values_dual.shape[-1]):
            dual[..., k] += _interpn(points, values_dual[..., k], xi_real, method=method,
                                     bounds_error=bounds_error, fill_value=fill_value, **kwargs)

    return DualNumber(primal, dual)

interpolate.interpn = interpn_override


from scipy import ndimage

# Monkey patch scipy.ndimage.gaussian_filter
_gaussian_filter = ndimage.gaussian_filter

def gaussian_filter_override(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0, **kwargs):
    # Case 1: No dual numbers at all
    if not any(isinstance(arg, DualNumber) for arg in [input, sigma] +
               (list(sigma) if isinstance(sigma, (list, tuple)) else [])):
        return _gaussian_filter(input, sigma, order=order, output=output, 
                                mode=mode, cval=cval, truncate=truncate, **kwargs)

    # Normalise sigma to one entry per axis; a scalar (possibly dual) sigma is shared by all axes,
    # and the partials of each axis are summed by the chain rule below.
    ndim = np.ndim(input.real if isinstance(input, DualNumber) else input)
    if isinstance(sigma, DualNumber) or not hasattr(sigma, '__iter__'):
        sigma_list = [sigma] * ndim
    else:
        sigma_list = list(sigma)
    sigma_real = np.array([s.real[0] if isinstance(s, DualNumber) else s for s in sigma_list], dtype=float)

    # Case 2: input is a DualNumber
    # gaussian_filter is linear, so it commutes with the dual components —
    # we can simply apply it independently to real and each dual slice.
    if isinstance(input, DualNumber):
        real = _gaussian_filter(input.real, sigma_real, order=order, output=output,
                                mode=mode, cval=cval, truncate=truncate, **kwargs)
        # dual shape: (*spatial, degree) — filter each degree slice independently
        degree = input.degree
        dual = np.stack([
            _gaussian_filter(input.dual[..., k], sigma_real, order=order,
                             mode=mode, cval=cval, truncate=truncate, **kwargs)
            for k in range(degree)
        ], axis=-1)
        filtered_input = DualNumber(real, dual)
    else:
        filtered_input = _gaussian_filter(input, sigma_real, order=order, output=output,
                                          mode=mode, cval=cval, truncate=truncate, **kwargs)

    # Case 3: sigma contains DualNumbers
    # d/d(sigma_i) [G_sigma * f] = (d/d(sigma_i) G_sigma) * f
    # The derivative of a Gaussian filter w.r.t. its sigma along axis i is computed
    # via finite differences of the filter itself.
    if any(isinstance(s, DualNumber) for s in sigma_list):
        if isinstance(filtered_input, DualNumber):
            degree = filtered_input.degree
            dual = filtered_input.dual.copy()
        else:
            degree = next(s.degree for s in sigma_list if isinstance(s, DualNumber))
            dual = np.zeros((*np.shape(filtered_input), degree))

        input_real = input.real if isinstance(input, DualNumber) else input

        for i, s in enumerate(sigma_list):
            if not isinstance(s, DualNumber):
                continue

            eps = np.sqrt(np.finfo(float).eps) * (1.0 + abs(s.real[0]))
            sigma_fwd = sigma_real.copy()
            sigma_bwd = sigma_real.copy()
            sigma_fwd[i] = max(sigma_real[i] + eps, 1e-10)  # sigma must be positive
            sigma_bwd[i] = max(sigma_real[i] - eps, 1e-10)

            df_dsigma_i = (
                _gaussian_filter(input_real, sigma_fwd, order=order, mode=mode, 
                                 cval=cval, truncate=truncate, **kwargs) -
                _gaussian_filter(input_real, sigma_bwd, order=order, mode=mode, 
                                 cval=cval, truncate=truncate, **kwargs)
            ) / (2 * eps)  # shape: (*spatial)

            # s.dual shape: (1, degree) — chain rule accumulation
            dual += df_dsigma_i[..., None] * s.dual

        filtered_input = DualNumber(
            filtered_input.real if isinstance(filtered_input, DualNumber) else filtered_input,
            dual
        )

    return filtered_input

ndimage.gaussian_filter = gaussian_filter_override