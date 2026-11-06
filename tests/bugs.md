# Known bugs and limitations

Reproduced on 2026-11-04 with numpy 1.26.4 and scipy 1.15.3 (the `trapz` error also on numpy 2.4.6). None of these are fixed.

Many only give a silently wrong answer for particular shapes, usually when an array's length equals the number of dual
variables, and raise an error otherwise. In the tables, `x`, `y` are `DualNumber`s:

```python
x = DualNumber([1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]])
y = DualNumber([2.0, 4.0], [[5.0, 6.0], [7.0, 8.0]])
arr = np.array([2.0, 4.0])
```

## Unsupported numpy calls are silently wrong

`__array_ufunc__` ignores `method` and `kwargs`, and numpy functions that are not monkey patched fall through to
numpy's own handling of an unknown object. Ufuncs that are not registered do raise.

| Call | Result |
| --- | --- |
| `np.min(x)`, `np.amax(x)`, `np.maximum.reduce(x)` | `x`, unreduced |
| `np.divide(x, arr, where=mask, out=out)` | `where` and `out` dropped |
| `np.nan_to_num(x)` | `x`, NaNs kept |
| `np.linalg.norm(x)`, `np.average(x)` | `x`, unchanged |
| `np.dot(x, x)` | element-wise product, not the inner product |
| `np.dot(arr, x)`, `np.cumsum(x)` | object array of `DualNumber`s |

## Singular derivatives

The chosen behaviour is below. Callers that hit these points should handle the derivative themselves.

| Call | Value of `.dual` |
| --- | --- |
| `np.sqrt` at 0, `np.log` at 0, `np.arccos` at 1 | `inf` for a nonzero incoming derivative, `nan` if it is exactly 0 |
| `np.arctan2` at the origin | 0 (undefined; arguably should be `nan`) |
| `abs(x)` at 0 | 0 |

## Wrong values

| Call | Got | Correct |
| --- | --- | --- |
| `np.interp(DualNumber([0.5, 1.5], ...), [0, 1, 2], [0, 10, 12]).real` (one interval used for all points) | `[5, 15]` | `[5, 11]` |
| `np.interp(DualNumber([-1.0], ...), [0, 1], [5, 15], left=0).real` | `-5` | `0`, or `5` without `left` |
| `x /= arr` (`.dual`) | `[[0.5, 0.5], [1.5, 1]]` | `[[0.5, 1], [0.75, 1]]` |
| `x /= y`, `np.divide(x, y)` (`.dual`) | `[[-0.75, -0.25], [-0.25, 0]]` | `[[-0.75, -0.5], [-0.125, 0]]` |
| `x ** arr` (`.dual`) | `[[2, 8], [12, 128]]` | `[[2, 4], [96, 128]]` |
| `x ** y` (`.dual`) | `[[2, 130.5], [6, 216.7]]` | `[[2, 4], [173.6, 216.7]]` |
| `DualNumber([3.0, 5.0], x.dual) // y` (`.dual`) | all `-4` | all `0` |
| `DualNumber([3.0, 5.0], x.dual) % y` (`.dual`) | `[[1, 2], [3, 4]]` | all `-4` |
| `DualNumber([3.0, 5.0], x.dual) // 2` (`.dual`) | `[[1, 2], [3, 4]]` | all `0` |
| `RectBivariateSpline(g, g, g[:, None] * g)(DualNumber([1.0, 2.0], [[1.0], [10.0]]), DualNumber([1.0, 2.0], [[0.0], [0.0]]))` with `grid=True`, `g = np.arange(4.0)` (`.dual`) | `[[1, 20], [1, 20]]` | `[[1, 2], [10, 20]]` |
| `np.sum(DualVariables([np.array([1.0, 2.0, 3.0]), ...])[0])` (`.dual`; `DualNumber` does not broadcast `dual`) | `[[1, 0]]` | `[[3, 0]]` |
| `np.trapz(y2d, x1d, axis=0)` with `y2d.dual == y2d.real[..., None]`, `y2d.real = [[1, 2], [3, 4], [5, 6]]`, `x1d = [0, 1, 3]` (`.dual`) | `[6, 16]` | `[10, 13]` |
| `make_interp_spline([0, 1, 2, 3], DualNumber([0, 2, 4, 6], zeros))(DualNumber([1.5], [[1.0]]))` (`.dual`; the query point's derivative is dropped) | `0` | `2` |
| `a[..., 0] = dual` (writes to the degree axis) | wrong dual elements | spatial column |
| `np.diff(DualNumber([1.0, 2.0], [[1.0], [1.0]]), prepend=5)` (`.dual`) | `[[-4], [0]]` | `[[1], [0]]` |
| `np.pad(a, 1, constant_values=7)` (`.dual`) | `7` in the padding | `0` |
| `interp1d(..., bounds_error=False, fill_value=9)(outside)`, `interpn(..., fill_value=9)` with dual values (`.dual`) | `9` | `0` |
| `gaussian_filter(a, DualNumber([1.125], ...))` (the kernel radius jumps at this sigma) | `d/dsigma` of order 1000 | order 1 |

## Raises an error

| Call | Error |
| --- | --- |
| `np.gradient(dual_2d, dx, dy)` | `TypeError`; all spacings are passed to each single axis |
| `interpn(points, values, [dual_2d, array_2d])` | `ValueError`; reals are flattened, duals are not |
| `gaussian_filter(dual, np.array(1.0))` | `TypeError`; a 0-d array sigma, on the dual path only |
| `np.clip(dual, 0, 1, out)`, `np.clip(dual, min=0, max=1)` | `TypeError` |
| `np.trapz(dual_2d, x, axis=1)`, `np.trapezoid(...)` (numpy 2) | `ValueError`; `x` is not broadcast along `axis`. `scipy.integrate.trapezoid` works |

## Other behaviour that differs from numpy

- `np.max(dual, keepdims=True)` with `axis=None` ignores `keepdims`. Ties take the first maximum.
- `x // y`, `x % y` and `x.flatten()` share `dual` with `x`, where numpy would return a new array. `DualNumber(real, dual)`
  never copies its inputs, and `x + 0` and `x - 0` do copy.
- `isinstance(f, scipy.interpolate.interp1d)` raises `TypeError` after import, because `interp1d` is replaced by a function.
- `*_like` returns a `DualNumber` with zero derivatives.
- Importing dualitic replaces numpy and scipy attributes, so `np.sum` and the rest are no longer the original
  objects. Calls without a `DualNumber` are forwarded unchanged.
- The `RegularGridInterpolator` and `interpn` query point derivatives are finite differences, not exact. Their
  step is about 1.5e-8, and at a kink they return the average of the two slopes.

## Missing comments

These make a decision that the code does not explain:

- `DualVariables`, `DualNumber.__init__` (no 0-d values, no broadcast, no copy), `//` and `%`, `__abs__` at 0
- `true_divide`, `arccos`, `sqrt`, `power`, `log`, and the dead `np.min` and `np.nanmax` registrations, which are not ufuncs
- `interp`, `__setitem__`, `__pow__`, `np.max` ties and `keepdims`, `where`, `diff`, `gradient`, `pad`, `trapz`,
  `cumulative_trapezoid`, `interp1d`, `RectBivariateSpline`, `make_interp_spline` query point, `interpn` flattened output,
  `gaussian_filter` sigma step
