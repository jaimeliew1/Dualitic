"""Verification of the monkey-patched scipy routines against finite differences / analytic results."""

import numpy as np
import pytest
from scipy import integrate, interpolate, ndimage

from dualitic import DualNumber

DEGREE = 3
RNG = np.random.default_rng(0)


def _dual_field(shape, degree=DEGREE):
    """Random smooth-ish dual field: real part and `degree` dual components."""
    return DualNumber(RNG.normal(size=shape), RNG.normal(size=(*shape, degree)))


def _directional_fd(func, real_in, dual_in, eps=1e-6):
    """Central FD of func along each of the `degree` dual directions of the input."""
    cols = []
    for k in range(dual_in.shape[-1]):
        fp = func(real_in + eps * dual_in[..., k])
        fm = func(real_in - eps * dual_in[..., k])
        cols.append((fp - fm) / (2 * eps))
    return np.stack(cols, axis=-1)


class TestGaussianFilter:
    @pytest.mark.parametrize("mode", ["reflect", "constant"])
    def test_dual_input_matches_fd(self, mode):
        u = _dual_field((12, 15))
        sigma = [1.3, 0.8]
        out = ndimage.gaussian_filter(u, sigma, mode=mode)
        ref = ndimage.gaussian_filter(u.real, sigma, mode=mode)
        np.testing.assert_allclose(out.real, ref, rtol=1e-12)
        fd = _directional_fd(lambda a: ndimage.gaussian_filter(a, sigma, mode=mode), u.real, u.dual)
        np.testing.assert_allclose(out.dual, fd, atol=1e-7)

    def test_float_input_unchanged(self):
        a = RNG.normal(size=(9, 9))
        np.testing.assert_array_equal(
            ndimage.gaussian_filter(a, [1.0, 2.0]), ndimage.gaussian_filter(a, [1.0, 2.0])
        )
        assert not isinstance(ndimage.gaussian_filter(a, [1.0, 2.0]), DualNumber)

    def test_dual_sigma_list_matches_fd(self):
        a = RNG.normal(size=(20, 20))
        s0, s1 = 1.5, 2.2
        sigma = [DualNumber([s0], [[1.0, 0.0]]), DualNumber([s1], [[0.0, 1.0]])]
        out = ndimage.gaussian_filter(a, sigma)
        assert isinstance(out, DualNumber)
        np.testing.assert_allclose(out.real, ndimage.gaussian_filter(a, [s0, s1]), rtol=1e-12)
        eps = 1e-4  # sigma enters through a truncated kernel, so use a moderate step
        d0 = (ndimage.gaussian_filter(a, [s0 + eps, s1]) - ndimage.gaussian_filter(a, [s0 - eps, s1])) / (2 * eps)
        d1 = (ndimage.gaussian_filter(a, [s0, s1 + eps]) - ndimage.gaussian_filter(a, [s0, s1 - eps])) / (2 * eps)
        np.testing.assert_allclose(out.dual[..., 0], d0, atol=1e-4)
        np.testing.assert_allclose(out.dual[..., 1], d1, atol=1e-4)

    def test_dual_sigma_and_dual_input_chain_rule(self):
        u = _dual_field((16, 16), degree=2)
        sigma = [DualNumber([1.4], [[1.0, 0.0]]), 1.4]
        out = ndimage.gaussian_filter(u, sigma)
        eps = 1e-4
        # total derivative wrt parameter 0: d(filter)/d(sigma_0) * 1 + filter(u.dual[..., 0])
        d_sigma = (ndimage.gaussian_filter(u.real, [1.4 + eps, 1.4]) - ndimage.gaussian_filter(u.real, [1.4 - eps, 1.4])) / (2 * eps)
        expected0 = d_sigma + ndimage.gaussian_filter(u.dual[..., 0], [1.4, 1.4])
        expected1 = ndimage.gaussian_filter(u.dual[..., 1], [1.4, 1.4])
        np.testing.assert_allclose(out.dual[..., 0], expected0, atol=1e-4)
        np.testing.assert_allclose(out.dual[..., 1], expected1, atol=1e-8)

    def test_scalar_dual_sigma(self):
        """sigma passed as a single DualNumber (isotropic) should differentiate, not be silently dropped."""
        a = RNG.normal(size=(20, 20))
        s = 1.7
        out = ndimage.gaussian_filter(a, DualNumber([s], [[1.0]]))
        eps = 1e-4
        fd = (ndimage.gaussian_filter(a, s + eps) - ndimage.gaussian_filter(a, s - eps)) / (2 * eps)
        np.testing.assert_allclose(out.dual[..., 0], fd, atol=1e-4)


class TestInterpn:
    def setup_method(self):
        self.points = (np.linspace(0, 2, 5), np.linspace(-1, 1, 7), np.linspace(0, 3, 9))
        self.shape = tuple(len(p) for p in self.points)

    def _queries(self, n=40):
        lo = np.array([p[0] for p in self.points])
        hi = np.array([p[-1] for p in self.points])
        return lo + (hi - lo) * RNG.uniform(0.02, 0.98, size=(n, 3))

    def test_dual_values_exact(self):
        """interpn is linear in the values, so dual part must equal interpolating each dual slice."""
        v = _dual_field(self.shape)
        xi = self._queries()
        out = interpolate.interpn(self.points, v, xi, method="linear")
        np.testing.assert_allclose(out.real, interpolate.interpn(self.points, v.real, xi), rtol=1e-12)
        fd = _directional_fd(lambda a: interpolate.interpn(self.points, a, xi), v.real, v.dual)
        np.testing.assert_allclose(out.dual, fd, atol=1e-8)

    def test_dual_query_points_on_linear_field(self):
        """For f = a.x + b (linear), the exact gradient wrt the query point is a."""
        a = np.array([1.5, -2.0, 0.7])
        X = np.stack(np.meshgrid(*self.points, indexing="ij"), axis=-1)
        values = X @ a + 0.3
        xi = self._queries(10)
        xi_dual = [DualNumber(xi[:, d], np.eye(3)[d] * np.ones((len(xi), 1))) for d in range(3)]
        out = interpolate.interpn(self.points, values, xi_dual, method="linear", bounds_error=False, fill_value=0)
        np.testing.assert_allclose(out.real, xi @ a + 0.3, rtol=1e-8)
        np.testing.assert_allclose(out.dual, np.tile(a, (len(xi), 1)), atol=1e-5)

    def test_dual_query_points_nonlinear(self):
        X = np.stack(np.meshgrid(*self.points, indexing="ij"), axis=-1)
        values = np.sin(X[..., 0]) * np.cos(X[..., 1]) + X[..., 2] ** 2
        xi = self._queries(10)
        xi_dual = [DualNumber(xi[:, d], np.eye(3)[d] * np.ones((len(xi), 1))) for d in range(3)]
        out = interpolate.interpn(self.points, values, xi_dual, method="linear")
        fd = np.stack(
            [
                (interpolate.interpn(self.points, values, xi + e * np.eye(3)[d])
                 - interpolate.interpn(self.points, values, xi - e * np.eye(3)[d])) / (2 * e)
                for d, e in zip(range(3), [1e-6] * 3)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(out.dual, fd, atol=1e-4)

    def test_out_of_bounds_zero_derivative(self):
        v = _dual_field(self.shape)
        xi = np.array([[10.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        out = interpolate.interpn(self.points, v, xi, bounds_error=False, fill_value=0)
        np.testing.assert_array_equal(out.real[0], 0.0)
        np.testing.assert_array_equal(out.dual[0], 0.0)
        assert np.any(out.dual[1] != 0)

    def test_float_unchanged(self):
        v = RNG.normal(size=self.shape)
        out = interpolate.interpn(self.points, v, self._queries())
        assert not isinstance(out, DualNumber)


class TestMiscPatches:
    def test_arctan2_origin_has_zero_gradient(self):
        y = DualNumber([0.0], [[1.0, 0.0]])
        z = DualNumber([0.0], [[0.0, 1.0]])
        out = np.arctan2(z, y)
        assert out.real[0] == 0.0
        assert np.all(np.isfinite(out.dual)) and np.all(out.dual == 0.0)

    def test_arctan2_away_from_origin(self):
        y = DualNumber([1.0], [[1.0, 0.0]])
        z = DualNumber([2.0], [[0.0, 1.0]])
        out = np.arctan2(z, y)
        np.testing.assert_allclose(out.dual[0], [-2 / 5, 1 / 5])

    def test_like_functions_preserve_dual_type(self):
        a = _dual_field((4, 5), degree=2)
        for fn in (np.zeros_like, np.ones_like, np.empty_like):
            out = fn(a)
            assert isinstance(out, DualNumber)
            assert out.real.shape == (4, 5) and out.dual.shape == (4, 5, 2)
        np.testing.assert_array_equal(np.ones_like(a).real, 1.0)
        np.testing.assert_array_equal(np.ones_like(a).dual, 0.0)

    def test_preallocate_and_fill_keeps_derivatives(self):
        a = _dual_field((6, 3), degree=2)
        out = np.zeros_like(a)
        out[1:-1, :] = a[2:, :] - a[:-2, :]
        np.testing.assert_allclose(out.dual[1:-1], a.dual[2:] - a.dual[:-2])
        np.testing.assert_array_equal(out.dual[0], 0.0)

    def test_isinf_isfinite(self):
        a = DualNumber([1.0, np.inf, np.nan], np.zeros((3, 1)))
        np.testing.assert_array_equal(np.isinf(a), [False, True, False])
        np.testing.assert_array_equal(np.isfinite(a), [True, False, False])


class TestScalarAddSubCopies:
    """x + scalar returns a new array, like numpy."""

    @pytest.mark.parametrize(
        "op", [lambda x: x + 0, lambda x: 0 + x, lambda x: x - 0, lambda x: np.add(x, 0), lambda x: np.subtract(x, 0)]
    )
    def test_dual_not_shared(self, op):
        x = DualNumber(np.arange(3.0), np.ones((3, 2)))
        y = op(x)
        assert isinstance(y, DualNumber)
        np.testing.assert_array_equal(y.real, x.real)
        np.testing.assert_array_equal(y.dual, x.dual)
        y[0] = 5.0
        np.testing.assert_array_equal(x.real, np.arange(3.0))
        np.testing.assert_array_equal(x.dual, np.ones((3, 2)))


class TestNaNMatchesNumpy:
    A = np.array([np.nan, 1.0, 3.0, np.nan])
    B = np.array([1.0, np.nan, 2.0, np.nan])

    @pytest.mark.parametrize("ufunc", [np.maximum, np.minimum])
    def test_real_part_either_argument(self, ufunc):
        expected = ufunc(self.A, self.B)
        a = DualNumber(self.A, np.ones((4, 2)))
        b = DualNumber(self.B, np.zeros((4, 2)))
        for out in (ufunc(a, b), ufunc(a, self.B), ufunc(self.A, b)):
            np.testing.assert_array_equal(out.real, expected)

    @pytest.mark.parametrize("ufunc", [np.maximum, np.minimum])
    def test_derivative_is_nan_only_where_value_is_nan(self, ufunc):
        out = ufunc(DualNumber(self.A, np.ones((4, 2))), self.B)
        np.testing.assert_array_equal(np.isnan(out.dual).all(axis=-1), np.isnan(out.real))
        assert not np.isnan(out.dual[2]).any()

    def test_clip_propagates_nan(self):
        x = np.array([np.nan, -1.0, 0.5, 2.0])
        out = np.clip(DualNumber(x, np.ones((4, 1))), 0, 1)
        np.testing.assert_array_equal(out.real, np.clip(x, 0, 1))
        np.testing.assert_array_equal(out.dual[:, 0], [np.nan, 0, 1, 0])

    def test_nan_derivative_with_finite_value_is_still_discarded(self):
        """The wake models rely on clip discarding the derivative of sqrt(0)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            root = np.sqrt(DualNumber([0.0], [[0.0]]))
        assert np.isnan(root.dual).all()
        np.testing.assert_array_equal(np.clip(root, 0.5, 1.0).dual, [[0.0]])

    @pytest.mark.parametrize("ufunc", [np.maximum, np.minimum])
    def test_tie_takes_derivative_of_second_argument(self, ufunc):
        out = ufunc(DualNumber([1.0], [[1.0]]), 1.0)
        np.testing.assert_array_equal(out.dual, [[0.0]])

    def test_clip_keyword_bounds(self):
        out = np.clip(DualNumber([-1.0, 0.5], np.ones((2, 1))), a_min=0)
        np.testing.assert_array_equal(out.real, [0.0, 0.5])
        np.testing.assert_array_equal(out.dual[:, 0], [0.0, 1.0])


class TestQueryPointDerivativeAtGridEdge:
    """f = 3 x0 - 2 x1 + 1, so the query point gradient is exact wherever the point is inside the grid."""

    GRID = (np.linspace(0.0, 2.0, 5), np.linspace(-1.0, 1.0, 5))
    VALUES = 3 * GRID[0][:, None] - 2 * GRID[1][None, :] + 1

    @staticmethod
    def _dual_query(point):
        return [DualNumber([p], [[float(i == d) for i in range(2)]]) for d, p in enumerate(point)]

    def _interpolators(self, **kwargs):
        rgi = interpolate.RegularGridInterpolator(self.GRID, self.VALUES, **kwargs)
        return [rgi, lambda xi: interpolate.interpn(self.GRID, self.VALUES, xi, **kwargs)]

    @pytest.mark.parametrize(
        "point", [(0.0, 0.3), (2.0, 0.3), (1.1, -1.0), (1.1, 1.0), (0.0, -1.0), (2.0, 1.0), (1.1, 0.3)]
    )
    def test_slope_at_and_inside_bounds(self, point):
        for f in self._interpolators():
            np.testing.assert_allclose(f(self._dual_query(point)).dual, [[3.0, -2.0]], atol=1e-6)

    def test_constant_fill_outside_has_zero_slope(self):
        for f in self._interpolators(bounds_error=False, fill_value=9.0):
            out = f(self._dual_query((3.0, 0.3)))
            np.testing.assert_array_equal(out.real, [9.0])
            np.testing.assert_array_equal(out.dual, [[0.0, 0.0]])

    def test_nan_fill_outside_has_zero_slope(self):
        for f in self._interpolators(bounds_error=False, fill_value=np.nan):
            out = f(self._dual_query((1.0, 5.0)))
            assert np.isnan(out.real).all()
            np.testing.assert_array_equal(out.dual, [[0.0, 0.0]])

    @pytest.mark.parametrize("point", [(3.0, 0.3), (2.0, 0.3), (1.0, -4.0)])
    def test_extrapolation_keeps_edge_slope(self, point):
        for f in self._interpolators(bounds_error=False, fill_value=None):
            np.testing.assert_allclose(f(self._dual_query(point)).dual, [[3.0, -2.0]], atol=1e-6)


class TestPlainCallsForwardedUnchanged:
    """Importing dualitic must not change numpy/scipy for inputs without a DualNumber."""

    def test_interp1d_axis(self):
        y = np.array([[0.0, 1.0], [10.0, 11.0]])
        np.testing.assert_array_equal(interpolate.interp1d([0.0, 1.0], y, axis=0)(0.5), [5.0, 6.0])
        np.testing.assert_array_equal(interpolate.interp1d([0.0, 1.0], y, axis=1)(0.5), [0.5, 10.5])
        np.testing.assert_array_equal(interpolate.interp1d([0.0, 1.0], y, "linear", 0)(0.5), [5.0, 6.0])

    def test_cumulative_trapezoid_x_keyword(self):
        out = integrate.cumulative_trapezoid([1.0, 1.0, 1.0], x=[0.0, 1.0, 3.0])
        np.testing.assert_array_equal(out, [1.0, 3.0])
        out = integrate.cumulative_trapezoid(y=[1.0, 1.0], x=[0.0, 2.0], initial=0)
        np.testing.assert_array_equal(out, [0.0, 2.0])

    def test_clip_keywords(self):
        a = np.array([-1.0, 0.5, 2.0])
        np.testing.assert_array_equal(np.clip(a, a_min=0, a_max=None), [0.0, 0.5, 2.0])
        np.testing.assert_array_equal(np.clip(a, a_max=1, a_min=None), [-1.0, 0.5, 1.0])
        out = np.empty(3)
        np.clip(a, 0, 1, out)
        np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])

    def test_reshape_keywords(self):
        kw = "newshape" if np.lib.NumpyVersion(np.__version__) < "2.1.0" else "shape"
        assert np.reshape(np.arange(6), **{kw: (2, 3)}).shape == (2, 3)
        np.testing.assert_array_equal(np.reshape(np.arange(4), (2, 2), "F"), [[0, 2], [1, 3]])

    def test_positional_and_first_keyword_arguments(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(np.sum(a=a), 10.0)
        np.testing.assert_array_equal(np.mean(a=a), 2.5)
        np.testing.assert_array_equal(np.max(a=a), 4.0)
        np.testing.assert_array_equal(np.squeeze(a=a[None]), a)
        np.testing.assert_array_equal(np.zeros_like(a=a), np.zeros((2, 2)))
        assert np.sum(a, 0, int).dtype == int
        out = np.empty(2)
        np.max(a, 0, out)
        np.testing.assert_array_equal(out, [3.0, 4.0])
        np.concatenate([a, a], 0, out := np.empty((4, 2)))
        np.testing.assert_array_equal(out[2:], a)

    def test_interp_fp_keyword(self):
        assert np.interp(0.5, [0.0, 1.0], fp=[0.0, 2.0]) == 1.0

    def test_trapezoid_positional_dx(self):
        trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        assert trapezoid([1.0, 1.0], None, 3.0) == 3.0
        assert integrate.trapezoid([1.0, 1.0], None, 3.0) == 3.0

    def test_return_types(self):
        original = np.lib.NumpyVersion(np.__version__) < "2.0.0"
        expected = list if original else tuple
        assert isinstance(np.meshgrid([0.0], [1.0]), expected)
        assert isinstance(np.atleast_1d(1.0, 2.0), expected)
        assert isinstance(np.atleast_2d(1.0, 2.0), expected)
        assert isinstance(np.atleast_3d(1.0, 2.0), expected)

    def test_interpolator_options(self):
        grid = (np.linspace(0.0, 1.0, 6),)
        values = grid[0] ** 2
        rgi = interpolate.RegularGridInterpolator(grid, values, method="cubic", solver_args={})
        np.testing.assert_allclose(rgi([0.5], nu=(1,)), [1.0], atol=1e-10)
        spline = interpolate.make_interp_spline(grid[0], values, 3, None)
        np.testing.assert_allclose(spline(0.5), 0.25)

    def test_gaussian_filter_0d_sigma(self):
        a = np.arange(9.0).reshape(3, 3)
        out = ndimage.gaussian_filter(a, np.array(1.0))
        np.testing.assert_array_equal(out, ndimage.gaussian_filter(a, 1.0))

    def test_gradient_2d_dual_without_axis(self):
        f = np.arange(12.0).reshape(3, 4)
        out = np.gradient(DualNumber(f, np.stack([2 * f], axis=-1)))
        plain = np.gradient(f)
        assert type(out) is type(plain) and len(out) == 2
        for o, p in zip(out, plain):
            np.testing.assert_array_equal(o.real, p)
            np.testing.assert_array_equal(o.dual[..., 0], 2 * p)
