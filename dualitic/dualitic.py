import numpy as np
from scipy import special, integrate, interpolate

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

        if isinstance(self.real[0], DualNumber):
            breakpoint()

    @property
    def degree(self):
        """
        Returns the degree of the dual number. i.e. the number of dual variables.
        """
        return len(self.dual[-1])

    def __repr__(self):
        return f"DualNumber(degree={self.degree}, {self.real}, {self.dual})"

    def __iter__(self):
        return (DualNumber(r, d) for r, d in zip(self.real, self.dual))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return DualNumber(self.real[index], self.dual[(*index, slice(None))])
        else:
            return DualNumber(self.real[index], self.dual[(index, slice(None))])

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


def monkey_patch(to_override):
    MONKEY_PATCHED.append(to_override)

    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.to_override = to_override

        return wrapper

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
    dual = (xd * yr[..., None] - xr[..., None] * yd) / (
        xr[..., None] ** 2 + yr[..., None] ** 2
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
    dual = real[..., None] * (x1_dual * x2_real[..., None] / x1_real[..., None] + x2_dual * np.log(x1_real[..., None]))
    return DualNumber(real, dual)


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


@register_dual_ufunc(np.greater_equal)
def _(x1, x2, *args, **kwargs):
    if isinstance(x1, DualNumber):
        x1 = x1.real

    if isinstance(x2, DualNumber):
        x2 = x2.real

    return np.greater_equal(x1, x2, *args, **kwargs)


@register_dual_ufunc(np.max)
def _(x, axis=None, **kwargs):
    return DualNumber(np.max(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))


@register_dual_ufunc(np.min)
def _(x, axis=None, **kwargs):
    return DualNumber(np.min(x.real, axis=axis, **kwargs), np.zeros_like(x.dual))


@register_dual_ufunc(np.maximum)
def _(a, b, *args, **kwargs):
    assert isinstance(a, DualNumber)
    assert isinstance(b, float) or isinstance(b, int)
    indices = np.where(a.real < b)

    out = a
    out.real[indices] = b
    out.dual[indices] = 0
    return out


@register_dual_ufunc(np.minimum)
def _(a, b, *args, **kwargs):
    assert isinstance(a, DualNumber)
    assert isinstance(b, float) or isinstance(b, int)
    indices = np.where(a.real > b)

    out = a
    out.real[indices] = b
    out.dual[indices] = 0
    return out


@register_dual_ufunc(np.deg2rad)
def _(x, axis=None, **kwargs):
    return DualNumber(np.deg2rad(x.real), np.deg2rad(x.dual))


@register_dual_ufunc(np.rad2deg)
def _(x, axis=None, **kwargs):
    return DualNumber(np.rad2deg(x.real), np.rad2deg(x.dual))


@register_dual_ufunc(np.isnan)
def _(x, *args, **kwargs):
    return np.isnan(x.real, *args, **kwargs)


@register_dual_ufunc(np.nanmax)
def _(x, *args, **kwargs):
    breakpoint()
    return np.nanmax(x.real, *args, **kwargs)


@register_dual_ufunc(special.erf)
def _(x):
    real = special.erf(x.real)
    return DualNumber(real, 2 * x.dual / np.sqrt(2) * np.exp(-real[..., None] ** 2))


### Monkey patching

# Monkey patch np.clip
_clip = np.clip


def clip_override(x, a, b, **kwargs):
    if isinstance(x, DualNumber):
        return np.minimum(np.maximum(x, a), b)
    else:
        return _clip(x, a, b, **kwargs)


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


def squeeze_override(x, axis=None, **kwargs):
    if isinstance(x, DualNumber):
        return DualNumber(np.squeeze(x.real, -1), np.squeeze(x.dual, -2))
    else:
        return _squeeze(x, axis=axis, **kwargs)


np.squeeze = squeeze_override

# Monkey patch np.trapz
_trapz = np.trapz


def trapz_override(y, x=None, axis=-1, **kwargs):
    if isinstance(y, DualNumber):
        real = _trapz(y.real, x=x, axis=axis, **kwargs)
        if axis < 0:
            axis -= 1
        dual = _trapz(y.dual, x=x[..., None], axis=axis, **kwargs)
        return DualNumber(real, dual)
    else:
        return _trapz(y, x=x, axis=axis, **kwargs)


np.trapz = trapz_override

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

# Monkey patch scipy.interpolate.RectBivariateSpline

_RectBivariateSpline = interpolate.RectBivariateSpline


class RectBivariateSpline_override(_RectBivariateSpline):
    """
    Dual-number compatible RectBivariateSpline. Currently interpolates only real
    values onto a dual number-based interpolation grid.

    """

    def __init__(self, x, y, z, bbox=[None] * 4, kx=3, ky=3, s=0):
        return super().__init__(x, y, z, bbox=bbox, kx=kx, ky=ky, s=s)

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
