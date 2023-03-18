[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# gh-levy-simulation
A Python simulation algorithm for the generalised hyperbolic process based on point process simulation methods.

To use the simulator, import the `GeneralisedHyperbolic` class from the `generalisedhyperbolic.py` file:

```
from generalisedhyperbolic import GeneralisedHyperbolic
```

In order to initialise a `GeneralisedHyperbolic` object, the 5 arguments characterising the distribution are `lam`, `gamma`, `delta`, `beta`, `sigma`. By default the rate is defined as 1, optionally any `rate` value can be provided. To run the residual approximation method using a Brownian motion with drift, set `residual_mode='Gaussian'`.  

```
simulator = GeneralisedHyperbolic(lam=lam, gamma=gamma, delta=delta, beta=beta, sigma=sigma, residual_mode='Gaussian')
```

Jump magnitudes can be generated as

```
x_series = simulator.simulate_jumps()
```

where the corresponding jump times are 

```
t_series = np.random.uniform(low=0., high=1., size=x_series.size)
```

Some example code templates are provided in `simulation.ipynb`.
