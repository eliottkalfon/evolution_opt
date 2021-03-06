![](https://github.com/eliottkalfon/evolution_opt/blob/master/resources/eo_logo.png)

# Description

This package aims at providing a range of nature-inspired optimisation algorithms. 
The purpose of an optimisation algorithm is to find the maximum or minimum of a function. <br>
Genetic algorithms are particularly useful when it comes to **high-dimensional, non-linear and non-convex problems** (e.g. finding a needle in the 10-dimensional hay). 
They have a wide range of application from supply chain optimisation to hyperparameter tuning.
This first version includes an implementation of genetic algorithm with "regularized evolution".

Genetic algorithms are very useful in machine learning, especially in **hyperparameter tuning**.
The example folder contains two examples of genetic algorithms used to:<br>
1) Optimise the architecture and hyperparameters of a Neural Network ([link](https://github.com/eliottkalfon/evolution_opt/blob/master/example/Neural%20Network%20Optimisation.ipynb))<br>
2) Tune the hyperparameters of a Support Vector Machine and a XGBoost model ([link](https://github.com/eliottkalfon/evolution_opt/blob/master/example/SVM%20and%20XGBoost%20Optimisation.ipynb))

The full documentation can be found [here](https://eliottkalfon.github.io/evolution_opt/).

# Installation

This package can be installed with "pip" or by cloning this repository

    $ pip install evolution_opt
	
# Dependencies

To install and run evolution_opt make sure that you have installed the following packages

    $ pip install numpy pandas scipy matplotlib
	
# Importing evolution_opt

```python
import numpy as np
import pandas as pd
from evolution_opt.genetic import *
```

# Example Usage

## 1) Define a function to be optimised
    
This function has to take a dictionary of parameter as argument:
```python
def difficult_problem(param_dict):
    result = param_dict['x']**2 + (param_dict['y']+1)**2
    if param_dict['luck'] == 'lucky':
        pass
    else:
        result += 10
    return result
```
  This function could be **any process** that takes parameters as input and outputs a scalar value.
    
  It could evaluate a model's cross-validation score based on given hyperparameter values,
  a profit/cost function, the efficiency of a resourcing plan... The possibilities are limitless.
    
 ## 2) Define a search space
```python
search_space = [
    Integer(-100,100, 'x'),
    Real(-100,100, 'y'),
    Categorical(['lucky', 'unlucky'], 'luck')
]
```   
  The search space can be composed of Integer, Real and Categorical variables.
  Numeric parameters are initialised with a lower bound, upper bound and a parameter name.
  Categorical parameters require a list of possible values and a parameter name.
    
  ## 3) Run the evolutionary algorithm
```python
best_params = optimise(difficult_problem,search_space,minimize=True, 
                           population_size=20,n_rounds=500)   

# Prints:
# Number of Iterations: 500
# Best score: 0.00410559779230605
# Best parameters: {'x': -0.0, 'y': -1.0640749388786759, 'luck': 'lucky'}
```

# Credits

- Icon featured in the logo: Icon made by <a href="https://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon"> www.flaticon.com</a>
- Regularized Evolution Algorithm inspiration: Saltori, Cristiano, et al. ["Regularized Evolutionary Algorithm for Dynamic Neural Topology Search."](https://arxiv.org/abs/1905.06252) *International Conference on Image Analysis and Processing*. Springer, Cham, 2019.





