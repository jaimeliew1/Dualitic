# `Dualitic`-compatible functions
This document enumerates the Numpy and Scipy functions and classes incorporated into Dualitic, ensuring proper propagation of dual numbers in code employing these functions.


## Numpy
A comprehensive list of Numpy ans Scipy functions which are implemented (and not yet implemented) in Dualitic. Numpy [Universal functions](https://numpy.org/doc/stable/reference/ufuncs.html) (`ufunc`) are indicated with the ![](https://img.shields.io/badge/ufunc-blue) sheild.



### Trigonometric functions
| Function          | Implementation Status                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `sin(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `cos(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `tan(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `arcsin(x)`       | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `arccos(x)`       | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `arctan(x)`       | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `arctan2(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `hypot(x1, x2)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `degrees(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `radians(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `unwrap(x)`       | ![](https://img.shields.io/badge/Implemented-No-red)                                                 |
| `deg2rad(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `rad2deg(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |

### Hyperbolic functions
| Function     | Implementation Status                                                                             |
| ------------ | ------------------------------------------------------------------------------------------------- |
| `sinh(x)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `cosh(x)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `tanh(x)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `arcsinh(x)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `arccosh(x)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `arctanh(x)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
### Rounding
| Function    | Implementation Status                                                                             |
| ----------- | ------------------------------------------------------------------------------------------------- |
| `round(x)`  | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| `around(x)` | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| `rint(x)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `fix(x)`    | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| `floor(x)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `ceil(x)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `trunc(x)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |


### Sums, products, differences
| Function        | Implementation Status                                   |
| --------------- | ------------------------------------------------------- |
| `prod(x)`       | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `sum(x)`        | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `mean(x)`       | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `nanprod(x)`    | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `nansum(x)`     | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `cumprod(x)`    | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `cumsum(x)`     | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `nancumprod(x)` | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `nancumsum(x)`  | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `diff(x)`       | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (`prepend`/`append` give a wrong derivative) |
| `ediff1d(x)`    | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `gradient(x)`   | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (one spacing per call; several spacings raise) |
| `cross(x)`      | ![](https://img.shields.io/badge/Implemented-No-red)    |
| `trapz(x)`      | ![](https://img.shields.io/badge/Implemented-Yes-green) (also `trapezoid`, numpy >= 2) |
### Exponents and logarithms
| Function             | Implementation Status                                                                                |
| -------------------- | ---------------------------------------------------------------------------------------------------- |
| `exp(x)`             | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `expm1(x)`           | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `exp2(x)`            | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `log(x)`             | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `log10(x)`           | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `log2(x)`            | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `log1p(x)`           | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `logaddexp(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `logaddexp2(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
### Other special functions
| Function  | Implementation Status                                |
| --------- | ---------------------------------------------------- |
| `i0(x)`   | ![](https://img.shields.io/badge/Implemented-No-red) |
| `sinc(x)` | ![](https://img.shields.io/badge/Implemented-No-red) |


### Floating point routines
| Function            | Implementation Status                                                                             |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| `signbit(x)`        | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `copysign(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `frexp(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `ldexp(x1, x2)`     | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `nextafter(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `spacing(x)`        | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |


### Rational routines
| Function      | Implementation Status                                                                             |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `lcm(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `gcd(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
### Arithmetic operations
| Function               | Implementation Status                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `add(x1, x2)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `reciprocal(x)`        | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `positive(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `negative(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `multiply(x1, x2)`     | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `divide(x1, x2)`       | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `power(x1, x2)`        | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Partial-yellow) (wrong derivative for an array or dual exponent) |
| `subtract(x1, x2)`     | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `true_divide(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Partial-yellow) (the `/` operator is correct; `np.divide` of two dual numbers and in-place `/=` give a wrong derivative) |
| `floor_divide(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `float_power(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `fmod(x1, x2)`         | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `mod(x1, x2)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `modf(x)`              | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `remainder(x1, x2)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `divmod(x1, x2)`       | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
### Handling complex numbers
| Function       | Implementation Status                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------- |
| `angle(x)`     | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `real(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `imag(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `conj(x)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| `conjugate(x)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |

### Extrema Finding
| Function          | Implementation Status                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `maximum(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green)       |
| `max(x)`          | ![](https://img.shields.io/badge/Implemented-Yes-green) (`keepdims` ignored with `axis=None`; ties take the first maximum) |
| `amax(x)`         | ![](https://img.shields.io/badge/Implemented-No-red)                                                       |
| `fmax(x1, x2)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)          |
| `nanmax(x)`       | ![](https://img.shields.io/badge/Implemented-No-red) (registered as a ufunc, but `nanmax` is not one, so it is never called) |
| `minimum(x1, x2)` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green)       |
| `min(x)`          | ![](https://img.shields.io/badge/Implemented-No-red)                                                       |
| `amin(x)`         | ![](https://img.shields.io/badge/Implemented-No-red)                                                       |
| `fmin(x1, x2)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)          |
| `nanmin(x)`       | ![](https://img.shields.io/badge/Implemented-No-red)                                                       |
### Miscellaneous
| Function                | Implementation Status                                                                                |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| `convolve(x, v)`        | ![](https://img.shields.io/badge/Implemented-No-red)                                                 |
| `clip(x, a_min, a_max)` | ![](https://img.shields.io/badge/Implemented-Yes-green)                                              |
| `sqrt(x)`               | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `cbrt(x)`               | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `square(x)`             | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `absolute(x)`           | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `fabs(x)`               | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `sign(x)`               | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (zero derivative) |
| `heaviside(x1, x2)`     | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)    |
| `nan_to_num(x)`         | ![](https://img.shields.io/badge/Implemented-No-red)                                                 |
| `real_if_close(x)`      | ![](https://img.shields.io/badge/Implemented-No-red)                                                 |
| `interp(x)`             | ![](https://img.shields.io/badge/Implemented-Partial-yellow)                                         |



### Other Numpy univeral funcs
| Function               | Implementation Status                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `bitwise_and(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `bitwise_or(x1, x2)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `bitwise_xor(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `invert(x)`            | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `left_shift(x1, x2)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `right_shift(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `greater(x1, x2)`      | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (compares the real part) |
| `greater_equal(x1, x2` | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (compares the real part) |
| `less(x1, x2)`         | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (compares the real part) |
| `less_equal(x1, x2)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (compares the real part) |
| `not_equal(x1, x2)`    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (compares the real part) |
| `equal(x1, x2)`        | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (compares the real part) |
| `logical_and(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `logical_or(x1, x2)`   | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `logical_xor(x1, x2)`  | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `logical_not(x)`       | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |
| `isfinite(x)`          | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (real part) |
| `isinf(x)`             | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green) (real part) |
| `isnan(x)`             | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-Yes-green)  |
| `isnat(x)`             | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red)     |



## Numpy Array creation
from https://numpy.org/doc/stable/reference/routines.array-creation.html

| Function                                  | Implementation Status                                        |
| ----------------------------------------- | ------------------------------------------------------------ |
| `empty(shape)`                            | ![](https://img.shields.io/badge/Implemented-N/A-grey)       |
| `empty_like(prototype)`                   | ![](https://img.shields.io/badge/Implemented-Yes-green)         |
| `eye(N)`                                  | ![](https://img.shields.io/badge/Implemented-N/A-grey)       |
| `identity(n)`                             | ![](https://img.shields.io/badge/Implemented-N/A-grey)       |
| `ones(shape)`                             | ![](https://img.shields.io/badge/Implemented-N/A-grey)       |
| `ones_like(a)`                            | ![](https://img.shields.io/badge/Implemented-Yes-green)         |
| `zeros(shape)`                            | ![](https://img.shields.io/badge/Implemented-N/A-grey)       |
| `zeros_like(a)`                           | ![](https://img.shields.io/badge/Implemented-Yes-green)         |
| `full(shape, fill_value)`                 | ![](https://img.shields.io/badge/Implemented-N/A-grey)       |
| `full_like(a, fill_value)`                | ![](https://img.shields.io/badge/Implemented-No-red)         |
| `meshgrid(*xi[, copy, sparse, indexing])` | ![](https://img.shields.io/badge/Implemented-Partial-yellow) |


## Numpy Array manipulation
from https://numpy.org/doc/stable/reference/routines.array-manipulation.html

| Function                     | Implementation Status |
| ---------------------------- | --------------------- |
| `reshape(a, newshape)`       | ![](https://img.shields.io/badge/Implemented-Yes-green) (also the `DualNumber.reshape` method) |
| `squeeze(a)`                 | ![](https://img.shields.io/badge/Implemented-Yes-green) (also the `DualNumber.squeeze` method) |
| `expand_dims(a, axis)`       | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `ndarray.flatten()`          | ![](https://img.shields.io/badge/Implemented-Yes-green) (`DualNumber.flatten`; shares memory with the input) |
| `concatenate(arrays)`        | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `stack(arrays)`              | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `hstack(tup)`                | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `vstack(tup)`                | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `broadcast_arrays(*args)`    | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `pad(array, pad_width)`      | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (`constant_values` is also written into the derivative) |
| `where(condition, x, y)`     | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `argsort(a)`                 | ![](https://img.shields.io/badge/Implemented-Yes-green) (sorts by the real part) |
| `ndarray.__setitem__`        | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (an `...` index addresses the derivative axis) |

## Numpy Fourier Transform
to do
from https://numpy.org/doc/stable/reference/routines.fft.html
## Numpy linear algebra
from https://numpy.org/doc/stable/reference/routines.linalg.html

### Matrix and vector products
| Function                                         | Implementation Status                                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| dot(a, b[, out])                                 | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| linalg.multi_dot(arrays, *[, out])               | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| vdot(a, b, /)                                    | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| inner(a, b, /)                                   | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| outer(a, b[, out])                               | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| matmul(x1, x2, /[, out, casting, order, ...])    | ![](https://img.shields.io/badge/ufunc-blue) ![](https://img.shields.io/badge/Implemented-No-red) |
| tensordot(a, b[, axes])                          | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| einsum(subscripts, *operands[, out, dtype, ...]) | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| einsum_path(subscripts, *operands[, optimize])   | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| linalg.matrix_power(a, n)                        | ![](https://img.shields.io/badge/Implemented-No-red)                                              |
| kron(a, b)                                       | ![](https://img.shields.io/badge/Implemented-No-red)                                              |

### Decompositions
| Function                                        | Implementation Status                                |
| ----------------------------------------------- | ---------------------------------------------------- |
| linalg.cholesky(a)                              | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.qr(a[, mode])                            | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.svd(a[, full_matrices, compute_uv, ...]) | ![](https://img.shields.io/badge/Implemented-No-red) |

### Matrix eigenvalues
| Function                   | Implementation Status                                |
| -------------------------- | ---------------------------------------------------- |
| linalg.eig(a)              | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.eigh(a[, UPLO])     | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.eigvals(a)          | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.eigvalsh(a[, UPLO]) | ![](https://img.shields.io/badge/Implemented-No-red) |

### Norms and other numbers
| Function                                     | Implementation Status                                |
| -------------------------------------------- | ---------------------------------------------------- |
| linalg.norm(x[, ord, axis, keepdims])        | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.cond(x[, p])                          | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.det(a)                                | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.matrix_rank(A[, tol, hermitian])      | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.slogdet(a)                            | ![](https://img.shields.io/badge/Implemented-No-red) |
| trace(a[, offset, axis1, axis2, dtype, out]) | ![](https://img.shields.io/badge/Implemented-No-red) |


### Solving equations and inverting matrices
| Function                           | Implementation Status                                |
| ---------------------------------- | ---------------------------------------------------- |
| linalg.solve(a, b)                 | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.tensorsolve(a, b[, axes])   | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.lstsq(a, b[, rcond])        | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.inv(a)                      | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.pinv(a[, rcond, hermitian]) | ![](https://img.shields.io/badge/Implemented-No-red) |
| linalg.tensorinv(a[, ind])         | ![](https://img.shields.io/badge/Implemented-No-red) |


## Scipy
Replaced in the `scipy` modules when `dualitic` is imported, so import `dualitic` before importing these names. Values are dual numbers unless noted. Known wrong results are listed in `tests/bugs.md`.

| Function                                    | Implementation Status |
| ------------------------------------------- | --------------------- |
| `integrate.trapezoid(y, x)`                 | ![](https://img.shields.io/badge/Implemented-Yes-green) |
| `integrate.cumulative_trapezoid(y, x)`      | ![](https://img.shields.io/badge/Implemented-Yes-green) (also `cumtrapz`) |
| `interpolate.interp1d(x, y)`                | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (dual `y` only; `fill_value` is also written into the derivative) |
| `interpolate.RectBivariateSpline(x, y, z)`  | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (dual query points; wrong derivative with `grid=True`) |
| `interpolate.RegularGridInterpolator`       | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (dual query points only, derivative by finite differences) |
| `interpolate.interpn`                       | ![](https://img.shields.io/badge/Implemented-Yes-green) (dual values or query points; query point derivative by finite differences) |
| `interpolate.make_interp_spline(x, y)`      | ![](https://img.shields.io/badge/Implemented-Partial-yellow) (dual `y` only; the query point derivative is dropped) |
| `ndimage.gaussian_filter(input, sigma)`     | ![](https://img.shields.io/badge/Implemented-Yes-green) (dual input or dual `sigma`) |
