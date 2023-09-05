import numpy as np
from scipy import special, integrate, interpolate

REGISTERED_DUAL_UFUNCS = {}


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


class DualNumber(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, real, dual):
        self.real = np.atleast_1d(real)
        self.dual = np.atleast_2d(dual)
        if isinstance(self.real[0], DualNumber):
            breakpoint()

    def __repr__(self):
        return f"DualNumber({self.real}, {self.dual})"

    def __iter__(self):
        return (DualNumber(r, d) for r, d in zip(self.real, self.dual))

    def __add__(self, other):
        if isinstance(other, DualNumber):
            real = self.real + other.real
            dual = self.dual + other.dual
        elif isinstance(other, np.ndarray):
            real = self.real + other
            dual = np.zeros_like(other[..., None]) + self.dual

        else:
            real = self.real + other
            dual = self.dual
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
            dual = self.dual
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in REGISTERED_DUAL_UFUNCS:
            return REGISTERED_DUAL_UFUNCS[ufunc](*inputs)
        else:
            raise NotImplementedError(f"{ufunc.__name__} not implemented")


def register_dual_ufunc(ufunc):
    def decorator(func):
        REGISTERED_DUAL_UFUNCS[ufunc] = func
        return func

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
    if isinstance(self, np.ndarray):
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


@register_dual_ufunc(np.sqrt)
def _(x):
    real = np.sqrt(x.real)
    # breakpoint()
    return DualNumber(real, x.dual / (2 * real[..., None]))


@register_dual_ufunc(np.exp)
def _(x):
    real = np.exp(x.real)
    return DualNumber(real, x.dual * real[..., None])


@register_dual_ufunc(np.log)
def _(x):
    return DualNumber(np.exp(x.real), x.dual / x.real[..., None])


@register_dual_ufunc(np.sum)
def _(x, axis=None, **kwargs):
    if axis is None:
        axis = tuple(range(x.real.ndim))
    return DualNumber(
        np.sum(x.real, axis=axis, **kwargs), np.sum(x.dual, axis=axis, **kwargs)
    )


@register_dual_ufunc(np.mean)
def _(x, axis=None, **kwargs):
    if axis is None:
        axis = tuple(range(x.real.ndim))
    return DualNumber(
        np.mean(x.real, axis=axis, **kwargs), np.mean(x.dual, axis=axis, **kwargs)
    )


@register_dual_ufunc(np.max)
def _(x, axis=None, **kwargs):
    return DualNumber(np.max(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))


@register_dual_ufunc(np.maximum)
def _(x, axis=None, **kwargs):
    return DualNumber(np.max(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))


@register_dual_ufunc(np.min)
def _(x, axis=None, **kwargs):
    return DualNumber(np.min(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))


@register_dual_ufunc(np.deg2rad)
def _(x, axis=None, **kwargs):
    return DualNumber(np.deg2rad(x.real), np.deg2rad(x.dual))


@register_dual_ufunc(np.rad2deg)
def _(x, axis=None, **kwargs):
    return DualNumber(np.rad2deg(x.real), np.rad2deg(x.dual))


@register_dual_ufunc(special.erf)
def _(x):
    real = special.erf(x.real)
    return DualNumber(real, 2 * x.dual / np.sqrt(2) * np.exp(-real[..., None] ** 2))


### Monkey patching

_squeeze = np.squeeze


def squeeze_override(x, axis=None, **kwargs):
    if isinstance(x, DualNumber):
        return DualNumber(np.squeeze(x.real, -1), np.squeeze(x.dual, -2))
    else:
        return _squeeze(x, axis=axis, **kwargs)


np.squeeze = squeeze_override

# Monkey patch cumtrapz
_cumtrapz = integrate.cumtrapz


def cumtrapz_override(x, *args, axis=-1, **kwargs):
    if isinstance(x, DualNumber):
        real = _cumtrapz(x.real, *args, axis=axis, **kwargs)
        if axis < 0:
            axis -= 1
        dual = _cumtrapz(x.dual, *args, axis=axis, **kwargs)
        return DualNumber(real, dual)
    else:
        return _cumtrapz(x, *args, axis=axis, **kwargs)


integrate.cumtrapz = cumtrapz_override


# Monkey patch scipy.interpolate.interp1d

_interp1d = interpolate.interp1d


def interp1d_override(x, y, *args, axis=-1, **kwargs):
    if isinstance(y, DualNumber):
        interp_real = _interp1d(x, y.real, *args, axis=axis, **kwargs)
        if axis < 0:
            axis -= 1
        interp_dual = _interp1d(x, y.dual, *args, axis=axis, **kwargs)
        return lambda x: DualNumber(interp_real(x), interp_dual(x))

    else:
        return _interp1d(x, y, *args, **kwargs)


interpolate.interp1d = interp1d_override
