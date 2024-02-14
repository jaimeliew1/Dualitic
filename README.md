# Dualitic

Dualitic is a Python package for forward mode automatic differentiation using dual numbers. 

## Key Features

- Seamlessly wrap NumPy and SciPy functions to enable automatic differentiation of any code. No need to rewrite models.
- Define dual variables that can be treated like normal NumPy arrays which track derivatives.
- Multivariate dual variables allow computing multiple derivatives at once.
- Dual arrays can be used with neural networks, optimization, and more.
- Lightweight and easy to use.

## Installation
To install, clone the repository and pip install.

```
git clone https://github.com/jaimeliew1/Dualitic.git
cd dualitic
pip install .
```
(Pypi installation comming soon)
## Usage

Import Dualitic and create a dual number by defining its real and dual part:

```python
import dualitic as dl

x = dl.DualNumber(3.0, [1.0]) 

print(x)
# DualNumber([3.], [[1.]])

print(x.real) 
# [3.]

print(x.dual)
# [[1.]] 
```

The `.real` attribute holds the primal value and `.dual` holds the derivatives.

Pass dual variables into NumPy/SciPy functions as normal:

```python
import numpy as np

y = np.sin(x)
print(y.real)
# [0.14112001]

print(y.dual)  
# [[-0.9899925]]
```

The derivative is computed automatically through the dual number propagation!

<!-- For multivariate derivatives, define multiple dual variables using `dl.DualVariables`:

```python

``` -->

See the documentation for more examples and API details. Dualitic enables derivative computation with only minor changes to existing code!

## Caveats
Dualitic can theoretically function with all Numpy and Scipy continuous functions, however not all methods have been implemented. See the [function list](function_list.md) for a comprehensive overview of implemented Numpy and Scipy functions, as well as functions which are not yet implemented.
<!-- 
## Documentation

Full documentation is available at [docs....](http://....).

## Contributing

Contributions to Dualitic are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License 

Dualitic is released under the MIT license. See [LICENSE](LICENSE) for details.

Let me know if you would like me to modify or expand the README in any way! I tried to cover the key points and motivate the use of Dualitic. -->