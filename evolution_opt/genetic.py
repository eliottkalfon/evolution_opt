#!/usr/bin/env python
# coding: utf-8

'''
This module is a Python implementation of a genetic algorithm with a regularized evolution process.
It was inspired by the following paper:

    Saltori, Cristiano, et al. "Regularized Evolutionary Algorithm for Dynamic Neural Topology Search." 
    International Conference on Image Analysis and Processing. Springer, Cham, 2019.
    
The example folder contains two examples of genetic algorithms used to:<br>
1) Optimise the architecture and hyperparameters of a Neural Network ([link](https://github.com/eliottkalfon/evolution_opt/blob/master/example/Neural%20Network%20Optimisation.ipynb))<br>
2) Tune the hyperparameters of a Support Vector Machine and XGBoost model ([link](https://github.com/eliottkalfon/evolution_opt/blob/master/example/SVM%20and%20XGBoost%20Optimisation.ipynb))


Example:
    To use this module, follow these three simple steps:
    
    1) Define a function to be optimised
    
    This function has to take a dictionary of parameter as argument:
    
        def difficult_problem(param_dict):
            result = param_dict['x']**2 + (param_dict['y']+1)**2
            if param_dict['luck'] == 'lucky':
                pass
            else:
                result += 10
            return result
    
    This function could be anything that takes parameters as input and outputs a scalar value.
    
    It could evaluate a model's cross-validation score based on given hyperparameter values,
    a profit/cost function, the efficiency of a resourcing plan... The possibilities are limitless.
    
    2) Define a search space
    
        search_space = [
            Integer(-100,100, 'x'),
            Real(-100,100, 'y'),
            Categorical(['lucky', 'unlucky'], 'luck')
        ]
        
    The search space can be composed of Integer, Real and Categorical variables.
    Numeric parameters are initialised with a lower bound, upper bound and a parameter name.
    Categorical parameters require a list of possible values and a parameter name.
    
    3) Run the evolutionary algorithm
    
        best_params = optimise(difficult_problem,search_space,minimize=True, 
                                   population_size=20,n_rounds=500)   
                                   
        # Prints:
        # Number of Iterations: 500
        # Best score: 0.00410559779230605
        # Best parameters: {'x': -0.0, 'y': -1.0640749388786759, 'luck': 'lucky'}
        
    
'''


import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import copy
import random


class Individual:
    '''
    Single individual in a population
    
    Args:
         id_number (int): identifier of the individual
         parent_id (tuple): tuple containing the id of the two parents
         stage_init (int): stage in which the individual was generated
         params (dict): dictionary of parameters
    
    Attributes:
        ind_id (int): identifier of the individual
        parent_id (tuple): tuple containing the id of the two parents
        stage_init (int): stage in which the individual was generated
        params (dict): dictionary of parameters
        fitness (float): fitness of the individual (default:None)
    '''
    def __init__(self, ind_id, parent_id, stage_init, params):
        self.ind_id = ind_id
        self.parent_id = parent_id
        self.stage_init = stage_init
        self.params = params
        self.fitness = None
    def get_fitness(self,opt_function):
        '''
        Evaluates an individual's fitness based on its parameters and a function
        
        Args:
            function (function): user-selected function to be optimised
        
        Note:
            This function must take a dictionary as argument. 
            This dictionary's keys must match the search space's parameter name
        
        Raises:
            ValueError if fitness cannot be evaluated for a given individual
        '''
        try:
            self.fitness = opt_function(self.params)
        except:
            raise ValueError("An error occurred while evaluating Individual {ind_id}'s fitness \n             with the following parameters: {params} \n             To debug this problem, please make sure that the following requirements are met: \n             1) The function requires a parameter as only required argument \n             2) The name of the dictionary keys expected by the function matches the parameter names             defined in the search space \n             3) The search space has been defined to avoid errors, or the function has been built to             handle them correctly \n             4) The function's execution should not generate errors".format(ind_id = self.ind_id, params = self.params))
    


class Integer():
    '''
    Integer Parameter class, member of the Search Space
    
    Args:
        lower_bound (int): parameter space lower bound
        upper_bound (int): parameter space upper bound
        name (str): parameter name
        step (int, optional): desired step between each selection
    
    Attributes:
        lower_bound (int): parameter space lower bound
        upper_bound (int): parameter space upper bound
        name (str): parameter name
        step (int, optional): desired step between each selection
        var_type (str): parameter type, used in the sampling process
        check (str): string 'parameter', used to check the integrity of the search space
    
    Note:
        This parameter's name must be consistent with the keys of the dictionary \
        fed into the optimised function
     
    Raises:
        ValueError if the lower bound is superior or equal to the lower bound
    '''
    def __init__(self, lower_bound, upper_bound, name, step = 1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
        self.step = step
        self.var_type = 'int'
        self.check = 'parameter'
        if self.lower_bound >= self.upper_bound:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(lower_bound, upper_bound))
        else:
            pass
        
class Real():
    '''
    Real Parameter class, member of the Search Space
    
    Args:
        lower_bound (int): parameter space lower bound
        upper_bound (int): parameter space upper bound
        name (str): parameter name
        precision (int, optional): desired number of decimals
        prior (str, optional): sampling distribution, default is 'uniform'. 'normal', 'lognormal',
        'exponential' and 'gamma' are also accepted
        mean (float, optional): default is 0, only required for 'normal' and 'lognormal' prior distributions
        stdev (float, optional): default is 1, only required for 'normal' and 'lognormal' prior distributions
        scale (float, oprional): default is 1, only required for 'exponential' and 'gamma' prior distributions
        shape (float, oprional): default is 1, only required for 'gamma' prior distributions
        
    
    Attributes:
        lower_bound (int): parameter space lower bound
        upper_bound (int): parameter space upper bound
        name (str): parameter name
        precision (int): desired number of decimals
        mean (float, optional): default is 0, only required for 'normal' and 'lognormal' prior distributions
        stdev (float, optional): default is 1, only required for 'normal' and 'lognormal' prior distributions
        scale (float, oprional): default is 1, only required for 'exponential' and 'gamma' prior distributions
        shape (float, oprional): default is 1, only required for 'gamma' prior distributions
        var_type (str): parameter type, used in the sampling process
        check (str): string 'parameter', used to check the integrity of the search space
    
    Note:
        This parameter's name must be consistent with the keys of the dictionary \
        fed into the optimised function
    
    Raises:
        ValueError if the lower bound is superior or equal to the lower bound
    '''
    def __init__(self, lower_bound, upper_bound, name,
                 precision = 3, prior = 'uniform',
                 mean = 0, stdev = 1, 
                 scale = 1, shape = 1):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
        self.precision = precision
        self.prior = prior
        self.mean = mean
        self.stdev = stdev
        self.scale = scale
        self.shape = shape
        self.var_type = 'real'
        self.check = 'parameter'
        if self.lower_bound >= self.upper_bound:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(lower_bound, upper_bound))
        else:
            pass
    def plot_prior(self):
        '''
        Plots the prior distribution of a Real parameter
        '''
        x = np.linspace(self.lower_bound, self.upper_bound, 200)
        print('Parameter Name: {name} \nPrior Distribution: {prior}'.format(name = self.name, prior = self.prior))
        if self.prior == 'uniform':
            y = stats.uniform.pdf(x)
        elif self.prior == 'normal':
            y = stats.norm.pdf(x, self.mean, self.stdev)
            print('Mean: {mean} \nStandard Deviation: {stdev}\n'.format(mean = self.mean, stdev = self.stdev))
        elif self.prior == 'lognormal':
            print('Mean: {mean} \nStandard Deviation: {stdev}\n'.format(mean = self.mean, stdev = self.stdev))
            y = stats.lognorm.pdf(x, s = self.stdev, scale = self.mean)
        elif self.prior == 'exponential':
            print('Scale: {scale} \n'.format(scale= self.scale))
            y = stats.expon.pdf(x, scale = self.scale)        
        elif self.prior == 'gamma':
            print('Shape: {shape} \nScale: {scale}'.format(shape = self.shape, scale = self.scale))
            y = stats.gamma.pdf(x, a = self.shape, scale = self.scale)
        plt.plot(x, y)
        plt.title(self.name, fontsize = 15, pad = 10)
        plt.show()
        
        
class Categorical():
    '''
    Categorical Parameter class, member of the Search Space
    
    Args:
        value_list (iterable): list of possible values
        name (str): parameter name
    
    Attributes:
        value_list (iterable): list of possible values
        name (str): parameter name
        var_type (str): parameter type, used in the sampling process
        check (str): string 'parameter', used to check the integrity of the search space
    
    Note:
        This parameter's name must be consistent with the keys of the dictionary
        fed into the optimised function
    
    Raises:
        ValueError if the value list argument is not an iterable
    '''
    def __init__(self, value_list, name):
        self.value_list = value_list
        self.name = name
        self.var_type = 'categorical'
        self.check = 'parameter'
        try:
            if isinstance(value_list, str):
                #Raises an error if the value_list is a string (separate check because strings are iterables...)
                raise ValueError("The list of possible values for the parameter '{}' cannot be a string"
                                 .format(self.name))
            iter(value_list)
        except: 
            #Raises an error if the value_list is not iterable
            raise ValueError("An iterable object has to be provided as list of values for the parameter '{}'"
                             .format(self.name))     


class Population:
    '''
    The PoPulation class is the main class of the genetic algorithm
    
    Args:
        pop_size (int): size of the population
        search_space (list): list of parameters initialised with the Integer, Real or Categorical classes defined above
        minimize (bool, default = True): True if the optimisation is a minimisation problem, False for maximisation
    
    Attributes:
        pop_size (int): size of the population
        search_space (list): list of parameters initialised with the Integer, Real or Categorical classes defined above
        stage (int): current evolution stage, set to 0
        id_count (int): individual counter, used to generate ids, set to 0
        reverse (bool): Logical negation of the minimize argument, used to determine the sort direction (ASC or DESC)
        param_names (list):list of the parameter names listed in the search space
        param_dict (dict): dictionary of parameters generated from the search space list

    Raises:
        ValueError if an item of the search space list is not a member of the Integer, Real or Categorical class
    '''
    def __init__(self, pop_size, search_space, minimize = True):
        self.pop_size = pop_size
        self.population = []
        self.stage = 0
        self.id_count = 0
        self.reverse = not minimize
        try:
            self.param_names = []
            self.param_dict = {}
            #Generates a list of parameter name and a parameter dictionary from the search space
            for param in search_space:
                self.param_names.append(param.name)
                self.param_dict[param.name] = param
        except:
             raise ValueError('Please make sure that the search space is a list of Integer,                               Real or Categorical parameter')
    
    def get_random_param(self, param_name):
        '''
        Randomly draws a parameter value
        
        Args:
            param_name (str): name of the parameter to be drawn
        
        Returns:
            A random parameter value selected from the search space
        '''
        if self.param_dict[param_name].var_type=='int':
            return random.randrange(self.param_dict[param_name].lower_bound,
                                    self.param_dict[param_name].upper_bound,
                                    self.param_dict[param_name].step)
        
        elif self.param_dict[param_name].var_type=='real':
            
            if self.param_dict[param_name].prior == 'uniform':
                return round(np.random.uniform(
                                     self.param_dict[param_name].lower_bound,
                                     self.param_dict[param_name].upper_bound
                                     ),self.param_dict[param_name].precision)
            elif self.param_dict[param_name].prior == 'normal':
                sample = round(np.random.normal(
                                     self.param_dict[param_name].mean,
                                     self.param_dict[param_name].stdev
                                     ),self.param_dict[param_name].precision)
                return min(max(self.param_dict[param_name].lower_bound, sample), self.param_dict[param_name].upper_bound)
                
            elif self.param_dict[param_name].prior == 'lognormal':
                sample = round(np.random.lognormal(
                                     self.param_dict[param_name].mean,
                                     self.param_dict[param_name].stdev
                                     ),self.param_dict[param_name].precision)
                return min(max(self.param_dict[param_name].lower_bound, sample), self.param_dict[param_name].upper_bound)
                
            elif self.param_dict[param_name].prior == 'gamma':
                sample = round(np.random.gamma(
                                     self.param_dict[param_name].shape,
                                     self.param_dict[param_name].scale
                                     ),self.param_dict[param_name].precision)
                return min(max(self.param_dict[param_name].lower_bound, sample), self.param_dict[param_name].upper_bound)
            
                
            elif self.param_dict[param_name].prior == 'exponential':
                sample = round(np.random.exponential(
                                     self.param_dict[param_name].scale
                                     ),self.param_dict[param_name].precision)
                return min(max(self.param_dict[param_name].lower_bound, sample), self.param_dict[param_name].upper_bound)
                
            
        elif self.param_dict[param_name].var_type=='categorical':
            return np.random.choice(self.param_dict[param_name].value_list)
        else:
            raise ValueError('Please make sure that the search space is a list of Integer(),                              Real() or Categorical() parameter')
        
    def get_initial_population(self):
        '''
        Randomly generates a population
        '''
        for idx in range(self.pop_size):
            temp_params = {}
            #Generates a random parameter value for each parameters in the parameter dictionary
            for i, param in enumerate(self.param_dict.keys()):
                temp_params[param] = self.get_random_param(param)
            self.population.append(Individual(ind_id = self.id_count,
                                              parent_id = None,    #no parent id as the population is generated
                                              stage_init = self.stage, 
                                              params = temp_params))
            #Increments the id_count by 1 as a new individual has been generated
            self.id_count +=1
            
    def evaluate_population(self,opt_function):
        '''
        Evaluates each of a population's individuals based on an optimisation function
        
        Args:
            opt_function (function): function to be optimised
        
        Note:
            To correctly define the optimisation function, please make sure that the following requirements are met:
            
            1) The function requires a parameter dictionary as only required argument
            
            2) The name of the dictionary keys expected by the function matches the parameter names
              defined in the search space
            
            3) The search space has been defined to avoid errors, or the function has been built to
              handle them correctly
            
            4) The function's execution should not generate errors
        '''
        for i,ind in enumerate(self.population):
            #Only evaluates individuals with no fitness
            #(i.e. that have not yet been evaluated)
            if self.population[i].fitness==None:
                self.population[i].get_fitness(opt_function)
            else:
                pass
            
    def sort_population(self):
        '''
        Sorts a population using its individual's fitness scores
        
        Note:
            This score will be ascending or descending based on the chosen direction of the optimisation problem
        '''
        self.population = sorted(self.population, key=lambda ind: ind.fitness, reverse=self.reverse)
        
    def natural_selection(self):
        '''
        Selects the n best individuals of a population
        This n is the population size
        '''
        self.sort_population()
        self.population = self.population[:self.pop_size]
        
    def get_offspring(self, n_children, n_sample, p_mutation, p_crossover):
        '''
        Generates a list of offspring
        
        Args:
            n_children (int): number of children
            n_sample (int): number of candidate parents sampled from the population
            p_mutation (float): mutation probability
            p_crossover (float): crossover probability
        '''
        #Increments the evolution stage by 1
        self.stage += 1
        children = []
        for child in range(n_children):
            #Draws a random sample of candidates from the population
            idx = np.random.randint(0, len(self.population), size=n_sample)
            candidates = sorted([self.population[i] for i in idx], key=lambda ind: ind.fitness, reverse=self.reverse)
            #Defines the best individual of the sample as parent 1
            p1 = candidates.pop(0)
            #Randomly selects parent 2 from the rest of the sample
            p2 = candidates[np.random.randint(0, len(candidates))]
            child_params = {}
            for i, param in enumerate(self.param_dict.keys()):
                #Theoretically speaking, mutation happens after crossover
                #but if a cell is mutated after crossover, the crossover operation is redundant
                #Mutation
                if np.random.uniform(0,1) < p_mutation:
                    child_params[param] = self.get_random_param(param)
                else:
                    #Crossover
                    if np.random.uniform(0,1) > p_crossover:
                        child_params[param] = p2.params[param]
                    else:
                        child_params[param] = p1.params[param]
            child = Individual(ind_id = self.id_count, 
                               parent_id = (p1.ind_id, p2.ind_id),
                               stage_init = self.stage, 
                               params = child_params)
            children.append(child)
            #Increments the id_count by 1 as a new individual has been generated
            self.id_count+=1
        self.population.extend(children)
        
    def round_log(self):
        '''
        Generates a round log, with a row for each individual id/stage combination
        
        Returns:
             list: evolution round description
        '''
        round_log = []
        for rank,individual in enumerate(self.population):
            params_list = []
            for param in self.param_names:
                params_list.append(individual.params[param])
            log_row = [str(self.stage) + '_' + str(individual.ind_id),    #stage_id individual identifier
                       self.stage,                                        #current stage
                       individual.ind_id,                                 #individual id
                       individual.parent_id,                              #parents' id
                       individual.stage_init,                             #stage in which the individual was generated
                       individual.fitness,                                #individual fitness
                       rank+1]                                            #rank within population at this given stage
            log_row.extend(params_list)                                   #list of parameter values
            round_log.append(log_row)
        return round_log
    
    def evolution(self, opt_function, n_rounds, n_children, n_sample, p_mutation, p_crossover, verbose):
        '''
        Executes the evolution algorithm for a set number of iterations
        
        Args:
             opt_function (function): function to be optimised
             n_rounds (int): number of evolution rounds
             n_children (int): number of children
             n_sample (int): number of candidate parents sampled from the population
             p_mutation (float): mutation probability
             p_crossover (float): crossover probability
             verbose (bool): prints progress if True
        
        Returns:
             dataframe: dataframe containing a summary of the evolution process, with a row by id/stage combination
        '''
        log_list = []
        for i in range(n_rounds):
            if verbose:
                print('Working on evolution round: {}'.format(i+1))
                print('Best score: {}'.format(self.population[0].fitness))
                print('Best parameters: {}'.format(self.population[0].params))
            self.get_offspring(n_children, n_sample, p_mutation, p_crossover)
            self.evaluate_population(opt_function)
            self.sort_population()
            log_list.extend(self.round_log())
            self.natural_selection()
        log_df = pd.DataFrame(data = log_list, columns=['index', 'stage', 'id', 'parent_id',
                                                        'stage_born', 'fitness', 'rank'] + self.param_names)
        log_df.set_index('index', inplace = True)
        return log_df
        
    def fitness_overtime(self,log_df):
        '''
        Plots fitness over time
        
        Args:
             dataframe: dataframe containing a summary of the evolution process
        '''
        log_df = log_df[log_df['rank']==1]
        ax = log_df.plot('stage', 'fitness') 
        ax.set_title('Fitness over time', fontsize = 20, pad=20)
        ax.set_xlabel('Generation (Program Iteration)', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
    
    def get_population_params(self):
        '''
        Generates a list of the population's parameters
        
        Returns:
            list: a list of the poulation's individuals' parameters
        '''
        param_list = []
        for ind in self.population:
            param_list.append(ind.params)
        return param_list
    
    def get_population_params_fitness(self):
        '''
        Generates a list of the population's parameters and their associated fitness
        
        Returns
            list: list of dictionaries containing each individual's parameters and fitness
        '''
        param_fitness = []
        for ind in self.population:
            param_fitness.append(
                {'parameters':ind.params, 
                 'fitness':ind.fitness}
            )
        return param_fitness
    
    def get_best_params(self):
        '''
        Outputs the best parameters of the population - i.e. the parameters of the population's first individual
        '''
        return self.population[0].params


def optimise(function, search_space,
             minimize=True, population_size=20,
             n_rounds=500, n_children=10, 
             n_sample=4, p_mutation=0.2, 
             p_crossover=0.6, verbose = False,
             return_log = False, return_population = False):
    
    '''
    Optimises a function given a search space
    
    Args:
        function (function): function to be optimised
        search_space (list): list of parameters, either Integer, Real or Categorical
        minimize (bool, default = True): nature of the optimsation objective
        population_size (int, default = 20): size of the population, number of individuals
        n_rounds (int, default = 500): number of evolution rounds
        n_children (int, default = 10): number of children generated at each round
        p_mutation (float, default = 0.2): probability of mutation
        p_crossover (float, default = 0.6): probability of crossover - i.e. taking the gene from the dominant parent
        verbose (bool, default = True): describes progress if True
        return_log (bool, default = False): returns an evolution log if true
        return_population (bool, default = False): returns a population object if true
    
    Returns:
        dict: parameters of the fittest individual after the final round of evolution
        dataframe, optional: dataframe containing a row for each individual id / stage combination
        Population, optional: Population object for customised analysis
    '''
    
    pop = Population(population_size, search_space=search_space, minimize = minimize)
    pop.get_initial_population()
    pop.evaluate_population(function)
    run_log = pop.evolution(function, n_rounds, n_children, n_sample, p_mutation, p_crossover, verbose)
    best_params = pop.get_best_params()  
    print('Number of Iterations: {}'.format(pop.stage))
    print('Best score: {}'.format(pop.population[0].fitness))
    print('Best parameters: {}'.format(pop.population[0].params))
    if return_log:
        if return_population:
            return best_params, run_log, pop
        else:
            return best_params, run_log
    else:
        if return_population:
            return best_params, pop
        else:
            return best_params


def solve(function, target_value, search_space, population_size=10,
             n_rounds=500, n_children=20, 
             n_sample=4, p_mutation=0.2, 
             p_crossover=0.6, verbose = False,
             return_log = False, return_population = False):
    '''
    Solves a function for a target value and a search space
    
    Args:
        function (function): function to be optimised
        target_value (float): function target value
        search_space (list): list of parameters, either Integer, Real or Categorical
        population_size (int, default = 20): size of the population, number of individuals
        n_rounds (int, default = 500): number of evolution rounds
        n_children (int, default = 10): number of children generated at each round
        p_mutation (float, default = 0.2): probability of mutation
        p_crossover (float, default = 0.6): probability of crossover - i.e. taking the gene from the dominant parent
        verbose (bool, default = True): describes progress if True
        return_log (bool, default = False): returns an evolution log if true
        return_population (bool, default = False): returns a population object if true
     
    Returns:
        dict: parameters of the fittest individual after the final round of evolution
        dataframe, optional: dataframe containing a row for each individual id / stage combination
        Population, optional: Population object for customised analysis
    '''
    def absolute_error(params):
        return abs(function(params)-target_value)
    
    pop = Population(population_size, search_space=search_space, minimize = True)
    pop.get_initial_population()
    pop.evaluate_population(absolute_error)
    run_log = pop.evolution(absolute_error, n_rounds, n_children, n_sample, p_mutation, p_crossover, verbose)
    best_params = pop.get_best_params()
    print('Number of Iterations: {}'.format(pop.stage))
    print('Lowest Absolute Error: {}'.format(pop.population[0].fitness))
    print('Best parameters: {}'.format(pop.population[0].params))
    if return_log:
        if return_population:
            return best_params, run_log, pop
        else:
            return best_params, run_log
    else:
        if return_population:
            return best_params, pop
        else:
            return best_params
    


def random_opt(function, search_space,
             minimize=True, n_iter = 20):
    
    '''
    Random search used for genetic algorithm benchmark
    
    Args:
        function (function): function to be optimised
        search_space (list): list of parameters, either Integer, Real or Categorical
        minimize (bool, default = True): nature of the optimsation objective
        n_iter (int): number of random evaluations
        
    Returns:
        dict: best set of parameters
        float: fitness associated with the best parameters
    '''

    
    pop = Population(n_iter, search_space=search_space, minimize = minimize)
    pop.get_initial_population()
    pop.evaluate_population(function)
    pop.sort_population()
    best_params = pop.population[0].params
    best_fitness = pop.population[0].fitness
    return best_params, best_fitness


def network_genealogy(log_df):
        '''
        Generates a networkX compatible genealogy in a pandas dataframe format
        
        Args:
            log_df (dataframe): pandas dataframe containing a summary of the evolution process
        
        Returns:
             dataframe: pandas dataframe convertible to a NetworkX graph
        '''
        #Removes duplicate ids by only taking the first instance of each individual
        log_df_unique = log_df.drop_duplicates(subset = ['id'] , keep = 'first')
        #Gets a unique list of parameters
        param_list = list(log_df)[list(log_df).index('rank')+1:]
        #List of columns in the genealogy dataframes
        column_list = ['parent', 'child', 'stage_born', 'fitness',
                       'child_rank', 'fitness','parent_type'] + param_list
        graph_list = []
        #Iterates through the rows of the unique individual dataframe
        for index, row in log_df_unique.iterrows():
            #If the individual has parents (i.e. was not individually generated)
            if row['parent_id']:
                for index, parent in enumerate(row['parent_id']):
                    row_list = [parent, row['id'], row['stage_born'], row['fitness'],
                                row['rank'], row['fitness'], index+1]
                    for param in param_list:
                        row_list.append(row[param])
                    graph_list.append(row_list)
            else:
                row_list = [row['parent_id'], row['id'], row['stage_born'], row['fitness'],
                                row['rank'], row['fitness'], None]
                for param in param_list:
                    row_list.append(row[param])
                graph_list.append(row_list)
                
        genealogy = pd.DataFrame(data = graph_list, 
                                 columns=column_list)
        
        return genealogy




