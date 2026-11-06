"""Tests for the ufuncs and numpy overrides added on the wes-extensions branch: tanh, sign, pad, __setitem__, squeeze, where
and diff. Known wrong results (see bugs.md) are marked xfail."""

from dualitic import DualNumber
import numpy as np
import pytest


def _vec():
    """DualNumber of shape (3,) with 2 dual variables."""
    return DualNumber(np.array([0.3, -1.2, 0.0]), np.array([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]]))


def _mat():
    """DualNumber of shape (2, 3) with 2 dual variables, dual[i, j] = (real[i, j], 10 real[i, j])."""
    real = np.arange(6.0).reshape(2, 3) + 1
    return DualNumber(real, np.stack([real, 10 * real], axis=-1))


class TestTanh:
    def test_value_and_derivative(self):
        x = _vec()
        out = np.tanh(x)
        assert np.allclose(out.real, np.tanh(x.real))
        assert np.allclose(out.dual, (1 - np.tanh(x.real) ** 2)[:, None] * x.dual)

    def test_finite_difference(self):
        x0, h = np.array([0.7]), 1e-6
        out = np.tanh(DualNumber(x0, np.array([[1.0]])))
        fd = (np.tanh(x0 + h) - np.tanh(x0 - h)) / (2 * h)
        assert np.allclose(out.dual[:, 0], fd)


class TestSign:
    def test_value_and_zero_derivative(self):
        x = _vec()
        out = np.sign(x)
        assert np.array_equal(out.real, [1.0, -1.0, 0.0])
        assert out.dual.shape == x.dual.shape
        assert np.all(out.dual == 0)


class TestPad:
    def test_plain_array_unchanged(self):
        a = np.arange(3.0)
        assert np.array_equal(np.pad(a, 1), np.pad(a.copy(), 1))
        assert not isinstance(np.pad(a, 1), DualNumber)

    def test_int_pad_width(self):
        x = _vec()
        out = np.pad(x, 1)
        assert np.array_equal(out.real, [0.0, 0.3, -1.2, 0.0, 0.0])
        assert out.dual.shape == (5, 2)
        assert np.array_equal(out.dual[1:-1], x.dual)
        assert np.all(out.dual[[0, -1]] == 0)

    def test_tuple_pad_width(self):
        x = _vec()
        out = np.pad(x, (1, 2))
        assert out.real.shape == (6,)
        assert out.dual.shape == (6, 2)
        assert np.array_equal(out.dual[1:4], x.dual)

    def test_per_axis_pad_width(self):
        x = _mat()
        out = np.pad(x, ((1, 0), (0, 2)))
        assert out.real.shape == (3, 5)
        assert out.dual.shape == (3, 5, 2)
        assert np.array_equal(out.dual[1:, :3], x.dual)
        assert np.all(out.dual[0] == 0) and np.all(out.dual[:, 3:] == 0)

    def test_edge_mode_copies_derivative(self):
        x = _vec()
        out = np.pad(x, 1, mode="edge")
        assert np.array_equal(out.real[[0, -1]], x.real[[0, -1]])
        assert np.array_equal(out.dual[0], x.dual[0])
        assert np.array_equal(out.dual[-1], x.dual[-1])

    @pytest.mark.xfail(reason="constant_values is also written into the dual padding (bugs.md)", strict=True)
    def test_constant_values_zero_derivative(self):
        out = np.pad(_vec(), 1, constant_values=7)
        assert np.all(out.real[[0, -1]] == 7)
        assert np.all(out.dual[[0, -1]] == 0)


class TestSetitem:
    def test_int_index_dual_value(self):
        x = _vec()
        x[1] = DualNumber(np.array([5.0]), np.array([[7.0, 8.0]]))
        assert x.real[1] == 5.0
        assert np.array_equal(x.dual[1], [7.0, 8.0])

    def test_slice_plain_value_zeros_derivative(self):
        x = _vec()
        x[1:] = 4.0
        assert np.array_equal(x.real, [0.3, 4.0, 4.0])
        assert np.array_equal(x.dual[0], [1.0, 0.0])
        assert np.all(x.dual[1:] == 0)

    def test_tuple_index_dual_value(self):
        x = _mat()
        x[1, 2] = DualNumber(np.array([-1.0]), np.array([[2.0, 3.0]]))
        assert x.real[1, 2] == -1.0
        assert np.array_equal(x.dual[1, 2], [2.0, 3.0])
        assert np.array_equal(x.dual[0, 0], [1.0, 10.0])

    def test_tuple_index_plain_value_zeros_derivative(self):
        x = _mat()
        x[0, :] = 0.0
        assert np.all(x.real[0] == 0)
        assert np.all(x.dual[0] == 0)
        assert np.array_equal(x.dual[1], _mat().dual[1])

    def test_size_matched_value_is_reshaped(self):
        # a shape-(1,) value into a scalar slot (numpy >= 2.4 raises without the reshape)
        x = _vec()
        x[0] = np.array([9.0])
        assert x.real[0] == 9.0

    @pytest.mark.xfail(reason="an Ellipsis index addresses the degree axis of dual (bugs.md)", strict=True)
    def test_ellipsis_index_dual_value(self):
        x = _mat()
        x[..., 0] = DualNumber(np.array([0.0, 0.0]), np.array([[1.0, 1.0], [2.0, 2.0]]))
        assert np.array_equal(x.dual[:, 0], [[1.0, 1.0], [2.0, 2.0]])


class TestSqueeze:
    def test_axis_none(self):
        x = DualNumber(np.ones((1, 3, 1)), np.ones((1, 3, 1, 2)))
        out = x.squeeze()
        assert out.real.shape == (3,)
        assert out.dual.shape == (3, 2)

    def test_axis_negative_keeps_degree_axis(self):
        x = DualNumber(np.ones((3, 1)), np.ones((3, 1, 1)))
        out = x.squeeze(axis=-1)
        assert out.real.shape == (3,)
        assert out.dual.shape == (3, 1)

    def test_axis_zero(self):
        x = DualNumber(np.ones((1, 3)), np.ones((1, 3, 2)))
        out = x.squeeze(axis=0)
        assert out.real.shape == (3,)
        assert out.dual.shape == (3, 2)


class TestWhere:
    def test_plain_arguments_unchanged(self):
        assert np.array_equal(np.where([True, False], [1, 2], [3, 4]), [1, 4])
        assert np.array_equal(np.where(np.array([0, 1, 1]))[0], [1, 2])

    def test_dual_and_plain(self):
        x = _vec()
        out = np.where(x.real > 0, x, np.zeros(3))
        assert np.array_equal(out.real, [0.3, 0.0, 0.0])
        assert np.array_equal(out.dual, [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    def test_plain_and_dual(self):
        # the degree comes from whichever argument is a DualNumber
        x = _vec()
        out = np.where(x.real > 0, np.ones(3), x)
        assert np.array_equal(out.real, [1.0, -1.2, 0.0])
        assert np.array_equal(out.dual, [[0.0, 0.0], [0.0, 2.0], [3.0, -1.0]])

    def test_dual_and_dual(self):
        x, y = _vec(), 2 * _vec()
        out = np.where([True, False, True], x, y)
        assert np.array_equal(out.dual, [[1.0, 0.0], [0.0, 4.0], [3.0, -1.0]])


class TestDiff:
    def test_last_axis(self):
        x = _mat()
        out = np.diff(x)
        assert np.array_equal(out.real, np.diff(x.real))
        assert np.array_equal(out.dual, np.diff(x.dual, axis=-2))

    def test_first_axis_second_order(self):
        x = DualNumber(np.array([1.0, 4.0, 9.0, 16.0]), np.array([[1.0], [2.0], [3.0], [4.0]]))
        out = np.diff(x, n=2)
        assert np.array_equal(out.real, [2.0, 2.0])
        assert np.all(out.dual == 0)

    @pytest.mark.xfail(reason="prepend is applied to dual as a value instead of zero derivative (bugs.md)", strict=True)
    def test_prepend_zero_derivative(self):
        out = np.diff(DualNumber(np.array([1.0, 2.0]), np.array([[1.0], [1.0]])), prepend=5)
        assert np.array_equal(out.dual, [[1.0], [0.0]])
