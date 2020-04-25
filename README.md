![](https://github.com/eliottkalfon/evolution_opt/blob/master/resources/eo_logo.png)

# Description

This package is a Python aims at providing a range of nature-inspired optimisation algorithms. This first version includes an implementation of genetic algorithm with "regularized evolution".

# Installation

This package can be installed with "pip" or by cloning this repository

    $ pip install evolution_opt

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


