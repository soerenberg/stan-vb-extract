[![Build Status](https://travis-ci.com/soerenberg/stan-vb-extract.svg?branch=main)](https://travis-ci.com/soerenberg/stan-vb-extract)
[![codecov](https://codecov.io/gh/soerenberg/stan-vb-extract/branch/main/graph/badge.svg?token=GI0ENVKQW5)](https://codecov.io/gh/soerenberg/stan-vb-extract)

# stan-vb-extract
Extract pystan vb samples combined with respect to parameter shapes.
The functions provided will extract variational inference samples from pystan
conveniently regarding the shapes of the parameters (as in the `extract` method
of the MCMC samples in pystan).

## Example
Consider the following Stan model

```{stan}
parameters {
    vector[3] a;
}
model {
    a ~ normal(0,1);
}
```

Then, pystan will return the samples of the example

```{python}
import pystan

stan_model = pystan.StanModel(...)

results = stan_model.vb(...)
```

In the format

```
results = {"a[1]": [...], "a[2]": [...], "a[3]": [...]}
```

The function `extract_vb_samples` will return the samples combined regarding
their shape. That is,

```{python}
samples = extract_vb_samples(results)
```

will return a dictionary `samples` where `samples["a"]` will be a numpy array
of shape `(num_samples, 3)` as it would have been found from MCMC samples in
pystan.
