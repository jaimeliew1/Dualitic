from dualitic import DualNumber, DualVariables
import numpy as np

class TestDual:
    def test_mean(self):
        a = DualNumber([1,2,3,4],[[1, 0],
                                  [1, 0],
                                  [1, 0],
                                  [1, 0]])
        a_mean = np.mean(a)
        print(a_mean.real)
        assert np.array_equal(a_mean.real, [2.5])
        print(a_mean.dual)
        assert np.array_equal(a_mean.dual, [[1, 0]])

    def test_mean_list_of_duals(self):
        a, b, c = DualVariables([1,2,3])
        list_mean = np.mean([a, b, c])
        assert np.array_equal(list_mean.real, [2])
        assert np.array_equal(list_mean.dual, [[1/3, 1/3, 1/3]])
        assert list_mean.degree == 3

    def test_sum(self):
        a = DualNumber([1,2,3,4],[[1, 0],
                                  [1, 0],
                                  [1, 0],
                                  [1, 0]])
        a_sum = np.sum(a)
        assert np.array_equal(a_sum.real, [10])
        assert np.array_equal(a_sum.dual, [[4, 0]])


class TestDualDual:
    def test_add_scalar(self):
        a = DualNumber(1, 2)
        b = DualNumber(3, 4)

        out = a + b
        assert out.real == 4
        assert out.dual == 6

        out = b + a
        assert out.real == 4
        assert out.dual == 6

    def test_add_vec(self):
        a = DualNumber([1, 2], [[3, 4, 5], [6, 7, 8]])
        b = DualNumber([10, 20], [[30, 40, 50], [60, 70, 80]])

        out = a + b
        assert np.array_equal(out.real, np.array([11, 22]))
        assert np.array_equal(out.dual, np.array([[33, 44, 55], [66, 77, 88]]))

        out = b + a
        assert np.array_equal(out.real, np.array([11, 22]))
        assert np.array_equal(out.dual, np.array([[33, 44, 55], [66, 77, 88]]))

    def test_div_vec(self):
        a = DualNumber([1, 2], [[3, 4, 5], [6, 7, 8]])
        b = DualNumber([10, 20], [[30, 40, 50], [60, 70, 80]])
        ######## TO DO TO DO TO DO!!!
        out = a + b
        assert np.array_equal(out.real, np.array([11, 22]))
        assert np.array_equal(out.dual, np.array([[33, 44, 55], [66, 77, 88]]))

        out = b + a
        assert np.array_equal(out.real, np.array([11, 22]))
        assert np.array_equal(out.dual, np.array([[33, 44, 55], [66, 77, 88]]))

    def test_meshgrid_2x2_dual_dual(self):
        x = DualNumber([1, 2], [[1, 2], [1, 2]])
        y = DualNumber([3, 4], [[5, 6], [7, 8]])

        xmesh, ymesh = np.meshgrid(x, y)

        assert np.array_equal(xmesh.real, [[1, 2], [1, 2]])
        assert np.array_equal(xmesh.dual, [[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
        assert np.array_equal(ymesh.real, [[3, 3], [4, 4]])
        assert np.array_equal(ymesh.dual, [[[5, 6], [5, 6]], [[7, 8], [7, 8]]])

    def test_meshgrid_3x2_dual_dual(self):
        x = DualNumber([1, 2, 3], [[1, 2], [1, 2], [1, 2]])
        y = DualNumber([3, 4], [[5, 6], [7, 8]])

        xmesh, ymesh = np.meshgrid(x, y)

        assert np.array_equal(xmesh.real, [[1, 2, 3], [1, 2, 3]])
        assert np.array_equal(
            xmesh.dual, [[[1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2]]]
        )
        assert np.array_equal(ymesh.real, [[3, 3, 3], [4, 4, 4]])
        assert np.array_equal(
            ymesh.dual, [[[5, 6], [5, 6], [5, 6]], [[7, 8], [7, 8], [7, 8]]]
        )

    def test_meshgrid_3x3_dual_dual(self):
        x = DualNumber([1, 2, 3], [[1, 2], [1, 2], [1, 2]])
        y = DualNumber([3, 4, 5], [[5, 6], [7, 8], [7, 8]])

        xmesh, ymesh = np.meshgrid(x, y)

        assert np.array_equal(xmesh.real, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert np.array_equal(
            xmesh.dual,
            [
                [[1, 2], [1, 2], [1, 2]],
                [[1, 2], [1, 2], [1, 2]],
                [[1, 2], [1, 2], [1, 2]],
            ],
        )
        assert np.array_equal(ymesh.real, [[3, 3, 3], [4, 4, 4], [5, 5, 5]])
        assert np.array_equal(
            ymesh.dual,
            [
                [[5, 6], [5, 6], [5, 6]],
                [[7, 8], [7, 8], [7, 8]],
                [[7, 8], [7, 8], [7, 8]],
            ],
        )


class TestDualScalar:
    def test_div_vector(self):
        a = DualNumber([1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]])
        b = 1

        out = b / a
        assert np.array_equal(out.real, np.array([1, 1 / 2, 1 / 3, 1 / 4]))
        assert np.array_equal(
            out.dual,
            np.array([[-1, -1], [-1 / 2, -1 / 2], [-1 / 3, -1 / 3], [-1 / 4, -1 / 4]]),
        )

    def test_flatten(self):
        a = DualNumber([[1, 2], [3, 4]], [[[1, 1], [2, 2]], [[3, 3], [4, 4]]])
        a_flat = a.flatten()
        assert np.array_equal([1, 2, 3, 4], a_flat.real)
        assert np.array_equal([[1, 1], [2, 2], [3, 3], [4, 4]], a_flat.dual)


class TestDualVec:
    def test_add_scalar(self):
        a = DualNumber(1, 2)
        b = np.array([1, 2, 3, 4])

        out = a + b
        assert np.array_equal(out.real, np.array([2, 3, 4, 5]))
        assert np.array_equal(out.dual, np.array([[2], [2], [2], [2]]))

        out = b + a
        assert np.array_equal(out.real, np.array([2, 3, 4, 5]))
        assert np.array_equal(out.dual, np.array([[2], [2], [2], [2]]))

    def test_sub_scalar(self):
        a = DualNumber(1, 2)
        b = np.array([1, 2, 3, 4])

        out = a - b
        assert np.array_equal(out.real, np.array([0, -1, -2, -3]))
        assert np.array_equal(out.dual, np.array([[2], [2], [2], [2]]))

        out = b - a
        assert np.array_equal(out.real, np.array([0, 1, 2, 3]))
        assert np.array_equal(out.dual, np.array([[-2], [-2], [-2], [-2]]))

    def test_mult_scalar(self):
        a = DualNumber(1, 2)
        b = np.array([1, 2, 3, 4])

        out = a * b
        assert np.array_equal(out.real, np.array([1, 2, 3, 4]))
        assert np.array_equal(out.dual, np.array([[2], [4], [6], [8]]))

        out = b * a
        assert np.array_equal(out.real, np.array([1, 2, 3, 4]))
        assert np.array_equal(out.dual, np.array([[2], [4], [6], [8]]))

    def test_div_scalar(self):
        a = DualNumber(1, 2)
        b = np.array([1, 2, 3, 4])

        out = a / b
        assert np.array_equal(out.real, np.array([1.0, 1 / 2, 1 / 3, 1 / 4]))
        assert np.array_equal(
            out.dual, np.array([[2.0], [1.0], [2.0 / 3.0], [1.0 / 2.0]])
        )

        out = b / a
        assert np.array_equal(out.real, np.array([1, 2, 3, 4]))
        assert np.array_equal(out.dual, np.array([[-2], [-4], [-6], [-8]]))

    def test_add_vec(self):
        a = DualNumber([1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]])
        b = np.array([1, 2, 3, 4])

        out = a + b
        assert np.array_equal(out.real, np.array([2, 4, 6, 8]))
        assert np.array_equal(out.dual, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))

        out = b + a
        assert np.array_equal(out.real, np.array([2, 4, 6, 8]))
        assert np.array_equal(out.dual, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))

    def test_sub_vec(self):
        a = DualNumber([1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]])
        b = np.array([1, 2, 3, 4])

        out = a - b
        assert np.array_equal(out.real, np.array([0, 0, 0, 0]))
        assert np.array_equal(out.dual, np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))

        out = b - a
        assert np.array_equal(out.real, np.array([0, 0, 0, 0]))
        assert np.array_equal(out.dual, -np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))

    def test_mult_vec(self):
        a = DualNumber([1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]])
        b = np.array([1, 2, 3, 4])

        out = a * b
        assert np.array_equal(out.real, np.array([1, 4, 9, 16]))
        assert np.array_equal(out.dual, np.array([[1, 1], [4, 4], [9, 9], [16, 16]]))

        out = b * a
        assert np.array_equal(out.real, np.array([1, 4, 9, 16]))
        assert np.array_equal(out.dual, np.array([[1, 1], [4, 4], [9, 9], [16, 16]]))

    def test_div_vec(self):
        a = DualNumber([1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]])
        b = np.array([1, 2, 3, 4])

        out = a / b
        assert np.array_equal(out.real, np.array([1, 1, 1, 1]))
        assert np.array_equal(out.dual, np.array([[1, 1], [1, 1], [1, 1], [1, 1]]))

        out = b / a
        assert np.array_equal(out.real, np.array([1, 1, 1, 1]))
        assert np.array_equal(out.dual, -np.array([[1, 1], [1, 1], [1, 1], [1, 1]]))

    def test_log_vec_self_consistent(self):
        theta = DualNumber(1, [[1.0]])
        xp = np.linspace(0, 10, 100) * theta

        assert np.array_equal(np.log(xp).real, np.log(xp.real))

    def test_erf_vec_self_consistent(self):
        from scipy.special import erf

        theta = DualNumber(1, [[1.0]])
        xp = np.linspace(0, 10, 100) * theta
        assert np.array_equal(erf(xp).real, erf(xp.real))

    def test_erf_grad(self):
        from scipy.special import erf

        theta = DualNumber(0, [[1.0]])
        xp = np.linspace(0, 10, 100) + theta
        dl_sol = erf(xp)
        an_deriv = (2 / np.sqrt(np.pi)) * np.exp(-xp.real**2)
        # breakpoint()
        assert np.array_equal(dl_sol.dual[:, 0], an_deriv)

    def test_log_grad(self):
        theta = DualNumber(0, [[1.0]])
        xp = np.linspace(0, 10, 100) + theta
        dl_sol = np.log(xp)
        an_deriv = 1 / xp.real
        assert np.array_equal(dl_sol.dual[:, 0], an_deriv)

    def text_exp_grad(self):
        theta = DualNumber(0, [[1.0]])
        xp = np.linspace(0, 10, 100) + theta
        dl_sol = np.exp(xp)
        an_deriv = dl_sol.real
        assert np.array_equal(dl_sol.dual[:, 0], an_deriv)

    def test_meshgrid_2x2_dual_vec(self):
        x = DualNumber([1, 2], [[3, 4], [5, 6]])
        y = np.array([7, 8])
        xmesh, ymesh = np.meshgrid(x, y)
        assert np.array_equal(xmesh.real, [[1, 2], [1, 2]])
        assert np.array_equal(xmesh.dual, [[[3, 4], [5, 6]], [[3, 4], [5, 6]]])
        assert np.array_equal(ymesh, [[7, 7], [8, 8]])

    def test_meshgrid_3x2_dual_vec(self):
        x = DualNumber([1, 2, 3], [[3, 4], [5, 6], [7, 8]])
        y = np.array([7, 8])
        xmesh, ymesh = np.meshgrid(x, y)
        assert np.array_equal(xmesh.real, [[1, 2, 3], [1, 2, 3]])
        assert np.array_equal(
            xmesh.dual, [[[3, 4], [5, 6], [7, 8]], [[3, 4], [5, 6], [7, 8]]]
        )
        assert np.array_equal(ymesh, [[7, 7, 7], [8, 8, 8]])

    def test_meshgrid_3x3_dual_vec(self):
        x = DualNumber([1, 2, 3], [[3, 4], [5, 6], [7, 8]])
        y = np.array([7, 8, 9])
        xmesh, ymesh = np.meshgrid(x, y)
        assert np.array_equal(xmesh.real, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert np.array_equal(
            xmesh.dual,
            [
                [[3, 4], [5, 6], [7, 8]],
                [[3, 4], [5, 6], [7, 8]],
                [[3, 4], [5, 6], [7, 8]],
            ],
        )
        assert np.array_equal(ymesh, [[7, 7, 7], [8, 8, 8], [9, 9, 9]])

    def test_meshgrid_2x2_vec_dual(self):
        x = [1, 2]
        y = DualNumber([3, 4], [[5, 6], [7, 8]])

        xmesh, ymesh = np.meshgrid(x, y)

        assert np.array_equal(xmesh, [[1, 2], [1, 2]])
        assert np.array_equal(ymesh.real, [[3, 3], [4, 4]])

        assert np.array_equal(ymesh.dual, [[[5, 6], [5, 6]], [[7, 8], [7, 8]]])

    def test_meshgrid_3x2_vec_dual(self):
        x = [1, 2, 3]
        y = DualNumber([3, 4], [[5, 6], [7, 8]])

        xmesh, ymesh = np.meshgrid(x, y)

        assert np.array_equal(xmesh, [[1, 2, 3], [1, 2, 3]])
        assert np.array_equal(ymesh.real, [[3, 3, 3], [4, 4, 4]])

        assert np.array_equal(
            ymesh.dual, [[[5, 6], [5, 6], [5, 6]], [[7, 8], [7, 8], [7, 8]]]
        )

    def test_meshgrid_3x3_vec_dual(self):
        x = [1, 2, 3]
        y = DualNumber([3, 4, 5], [[5, 6], [7, 8], [9, 0]])

        xmesh, ymesh = np.meshgrid(x, y)

        assert np.array_equal(xmesh, [[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert np.array_equal(ymesh.real, [[3, 3, 3], [4, 4, 4], [5, 5, 5]])
        assert np.array_equal(
            ymesh.dual,
            [
                [[5, 6], [5, 6], [5, 6]],
                [[7, 8], [7, 8], [7, 8]],
                [[9, 0], [9, 0], [9, 0]],
            ],
        )


def sample_function(x, y):
    return x * np.exp(-(x**2) - y**2)


def sample_function_dx(x, y):
    # Analytical x derivative of the sample function
    return (1 - 2 * x**2) * np.exp(-(x**2) - y**2)


def sample_function_dy(x, y):
    # Analytical y derivative of the sample function
    return -2 * x * y * np.exp(-(x**2) - y**2)


class TestMonkeyPatch:
    def test_RectBivariateSpline(self):
        from scipy.interpolate import RectBivariateSpline

        x = np.linspace(0, 1, 20)
        y = np.linspace(0, 1, 20)

        values = sample_function(*np.meshgrid(x, y))
        interpolator = RectBivariateSpline(x, y, values.T)

        _x = np.linspace(0, 1, 300)
        _y = np.linspace(0, 1, 300)
        xmesh, ymesh = np.meshgrid(_x, _y)
        values_interp = interpolator(xmesh, ymesh, grid=False)
        values_interp_dx = interpolator(xmesh, ymesh, dx=1, grid=False)
        values_interp_dy = interpolator(xmesh, ymesh, dy=1, grid=False)

        xmesh += DualNumber(0, [1, 0])
        ymesh += DualNumber(0, [0, 1])
        values_interp_dual = interpolator(xmesh, ymesh, grid=False)
        assert np.array_equal(values_interp_dual.real, values_interp)
        assert np.array_equal(values_interp_dual.dual[..., 0], values_interp_dx)
        assert np.array_equal(values_interp_dual.dual[..., 1], values_interp_dy)


class TestInterp:
    def test_interp_1(self):
        # test set 1: x, xp, and yp are all dual variables
        xp = DualNumber([1.1, 2.0], [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        yp = DualNumber([1.0, 3.0], [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
        x = DualNumber([1.5], [[0, 0, 0, 0, 1]])

        y_test1 = np.interp(x, xp, yp)

        x1_fd = 1.1
        y1_fd = 1
        x2_fd = 2
        y2_fd = 3
        x_fd = 1.5

        delta = 0.05
        dydx_fd = (
            np.interp(x_fd + delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd - delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydxl_fd = (
            np.interp(x_fd, [x1_fd + delta, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd, [x1_fd - delta, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydxr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd + delta], [y1_fd, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd - delta], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydyl_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd + delta, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd - delta, y2_fd])
        ) / (2 * delta)
        dydyr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd + delta])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd - delta])
        ) / (2 * delta)

        assert np.allclose(y_test1.dual[0, 0], dydxl_fd, 0.01)
        assert np.allclose(y_test1.dual[0, 1], dydxr_fd, 0.01)
        assert np.allclose(y_test1.dual[0, 2], dydyl_fd, 0.01)
        assert np.allclose(y_test1.dual[0, 3], dydyr_fd, 0.01)
        assert np.allclose(y_test1.dual[0, 4], dydx_fd, 0.01)
        assert np.allclose(
            y_test1.real[0], np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd]), 0.01
        )

    def test_interp_2(self):
        # test set 2: x is not a dual number while xp and yp are dual numbers
        xp = DualNumber([1.1, 2.0], [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        yp = DualNumber([1.0, 3.0], [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0]])
        x = 1.5

        y_test2 = np.interp(x, xp, yp)

        x1_fd = 1.1
        y1_fd = 1
        x2_fd = 2
        y2_fd = 3
        x_fd = 1.5

        delta = 0.05
        dydx_fd = (
            np.interp(x_fd + delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd - delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydxl_fd = (
            np.interp(x_fd, [x1_fd + delta, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd, [x1_fd - delta, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydxr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd + delta], [y1_fd, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd - delta], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydyl_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd + delta, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd - delta, y2_fd])
        ) / (2 * delta)
        dydyr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd + delta])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd - delta])
        ) / (2 * delta)

        assert np.allclose(y_test2.dual[0, 0], dydxl_fd, 0.01)
        assert np.allclose(y_test2.dual[0, 1], dydxr_fd, 0.01)
        assert np.allclose(y_test2.dual[0, 2], dydyl_fd, 0.01)
        assert np.allclose(y_test2.dual[0, 3], dydyr_fd, 0.01)
        assert np.allclose(
            y_test2.real[0], np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd]), 0.01
        )

    def test_interp_3(self):
        # test set 3: xp is not a dual number but x and yp are
        xp = [1.1, 2.0]
        yp = DualNumber([1.0, 3.0], [[1, 0, 0], [0, 1, 0]])
        x = DualNumber([1.5], [[0, 0, 1]])

        y_test3 = np.interp(x, xp, yp)

        x1_fd = 1.1
        y1_fd = 1
        x2_fd = 2
        y2_fd = 3
        x_fd = 1.5

        delta = 0.05
        dydx_fd = (
            np.interp(x_fd + delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd - delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydyl_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd + delta, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd - delta, y2_fd])
        ) / (2 * delta)
        dydyr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd + delta])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd - delta])
        ) / (2 * delta)

        assert np.allclose(y_test3.dual[0, 0], dydyl_fd, 0.01)
        assert np.allclose(y_test3.dual[0, 1], dydyr_fd, 0.01)
        assert np.allclose(y_test3.dual[0, 2], dydx_fd, 0.01)
        assert np.allclose(
            y_test3.real[0], np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd]), 0.01
        )

    def test_interp_4(self):
        # test set 4: yp is not a dual number but x and xp are
        xp = DualNumber([1.1, 2.0], [[1, 0, 0], [0, 1, 0]])
        yp = [1.0, 3.0]
        x = DualNumber([1.5], [[0, 0, 1]])

        y_test4 = np.interp(x, xp, yp)

        x1_fd = 1.1
        y1_fd = 1
        x2_fd = 2
        y2_fd = 3
        x_fd = 1.5

        delta = 0.05
        dydx_fd = (
            np.interp(x_fd + delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd - delta, [x1_fd, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydxl_fd = (
            np.interp(x_fd, [x1_fd + delta, x2_fd], [y1_fd, y2_fd])
            - np.interp(x_fd, [x1_fd - delta, x2_fd], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydxr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd + delta], [y1_fd, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd - delta], [y1_fd, y2_fd])
        ) / (2 * delta)
        dydyl_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd + delta, y2_fd])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd - delta, y2_fd])
        ) / (2 * delta)
        dydyr_fd = (
            np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd + delta])
            - np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd - delta])
        ) / (2 * delta)

        assert np.allclose(y_test4.dual[0, 0], dydxl_fd, 0.01)
        assert np.allclose(y_test4.dual[0, 1], dydxr_fd, 0.01)
        assert np.allclose(y_test4.dual[0, 2], dydx_fd, 0.01)
        assert np.allclose(
            y_test4.real[0], np.interp(x_fd, [x1_fd, x2_fd], [y1_fd, y2_fd]), 0.01
        )

    def test_interp_large_array(self):
        theta = DualNumber(1, [[1.0]])
        xp = np.linspace(0, 10, 100) * theta
        yp = xp**2

        x = np.linspace(0, 10, 5)

        y = np.interp(x, xp, yp)
        y_real = np.interp(x, xp.real, yp.real)

        assert np.allclose(y.real, y_real)
