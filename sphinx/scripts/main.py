#!/usr/bin/env python
# coding: utf-8

# # Genetic Algorithm Main Functions

# **To do:** <br>
# 1) Finish documentation <br>
# 2) Test error handling <br>
# 
# **Backlog:** <br>
# 1) Prior Distributions <br>
# 2) Constraints <br>
# 3) Early Stopping <br>
# 4) Plateau Avoidance <br>
# 5) Crowding Distance <br>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from sklearn.preprocessing import StandardScaler


# In[2]:


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
        
        Arguments
        function - user-selected function to be optimised
        
        Note
        This function must take a dictionary as argument. 
        This dictionary's keys must match the search space's parameter name
        '''
        try:
            self.fitness = opt_function(self.params)
        except:
            raise Exception("An error occurred while evaluating Individual {ind_id}'s fitness \n             with the following parameters: {params} \n             To debug this problem, please make sure that the following requirements are met: \n             1) The function requires a parameter as only required argument \n             2) The name of the dictionary keys expected by the function matches the parameter names             defined in the search space \n             3) The search space has been defined to avoid errors, or the function has been built to             handle them correctly \n             4) The function's execution should not generate errors".format(ind_id = self.ind_id, params = self.params))
    


# In[3]:


class Integer():
    '''
    Integer Parameter class, member of the Search Space
    
    Required inputs for initialisation
    lower_bound (int) - parameter space lower bound
    upper_bound (int) - parameter space upper bound
    name (str) - parameter name
    
    Attributes
    lower_bound (int) - parameter space lower bound
    upper_bound (int) - parameter space upper bound
    name (str) - parameter name
    var_type (str) - parameter type, used in the sampling process
    check (str) - string 'parameter', used to check the integrity of the search space
    
    Note
    This parameter's name must be consistent with the keys of the dictionary
    fed into the optimised function
    An error will also be raised if the lower bound is superior or equal to the lower bound
    '''
    def __init__(self, lower_bound, upper_bound, name):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
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
    
    Required inputs for initialisation
    lower_bound (int) - parameter space lower bound
    upper_bound (int) - parameter space upper bound
    name (str) - parameter name
    
    Attributes
    lower_bound (int) - parameter space lower bound
    upper_bound (int) - parameter space upper bound
    name (str) - parameter name
    var_type (str) - parameter type, used in the sampling process
    check (str) - string 'parameter', used to check the integrity of the search space
    
    Note
    This parameter's name must be consistent with the keys of the dictionary
    fed into the optimised function
    An error will also be raised if the lower bound is superior or equal to the lower bound
    '''
    def __init__(self, lower_bound, upper_bound, name):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.name = name
        self.var_type = 'real'
        self.check = 'parameter'
        if self.lower_bound >= self.upper_bound:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(lower_bound, upper_bound))
        else:
            pass
        
class Categorical():
    '''
    Categorical Parameter class, member of the Search Space
    
    Required inputs for initialisation
    value_list (iterable) - list of possible values
    name (str) - parameter name
    
    Attributes
    value_list (iterable) - list of possible values
    name (str) - parameter name
    var_type (str) - parameter type, used in the sampling process
    check (str) - string 'parameter', used to check the integrity of the search space
    
    Note
    This parameter's name must be consistent with the keys of the dictionary
    fed into the optimised function
    An error will also be raised if the lower bound is superior or equal to the lower bound
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


# In[4]:


class Population:
    '''
    The PoPulation class is the main class of the genetic algorithm
    
    Required inputs for initialisation
    pop_size (int) - size of the population
    search_space (list) - list of parameters initialised with the Integer, Real or Categorical classes defined above
    Optional
    minimize (bool) - True if the optimisation is a minimisation problem, False for maximisation (default: True)
    
    Attributes
    pop_size (int) - size of the population
    search_space (list) - list of parameters initialised with the Integer, Real or Categorical classes defined above
    stage (int) - current evolution stage, set to 0
    id_count (int) - individual counter, used to generate ids, set to 0
    reverse (bool) - Logical negation of the minimize argument, used to determine the sort direction (ASC or DESC)
    param_names (list) - list of the parameter names listed in the search space
    param_dict (dict) - dictionary of parameters generated from the search space list
    
    Methods
    get_random_param - randomly draws a new parameter value
    get_initial_population - randomly generates a new population
    evaluate_population - evaluates a population's fitness
    sort_population - sort a population based on its fitness
    natural_selection - selects the n best individuals of a population
    get_offspring - generates offspring and appends them to the population
    round_log - generates a log describing the evolution round's history
    evolution - executes the evolution algorithm
    fitness_overtime - plots the best population fitness by evolution round
    network_genealogy - returns a networkx compatible genealogy
    get_population_params - returns a list of the population's parameters
    get_population_params_fitness - returns a list of the population's parameters and fitness
    get_best_params - returns the best parameters of the population (based on fitness)
    
    Note
    An error will be raised if an item of the search space list is not a member of the Integer, Real or Categorical class
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
        
        Arguments
        param_name (str) - name of the parameter to be drawn
        
        Output
        Returns a random parameter value
        '''
        if self.param_dict[param_name].var_type=='int':
            return round(np.random.uniform(self.param_dict[param_name].lower_bound,
                                           self.param_dict[param_name].upper_bound),0)
        elif self.param_dict[param_name].var_type=='real':
            return np.random.uniform(self.param_dict[param_name].lower_bound,
                                     self.param_dict[param_name].upper_bound)
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
        
        Argument
        opt_function (function) - function to be optimised
        
        Note
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
        
        Note
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
        
        Arguments
        n_children (int) - number of children
        n_sample (int) - number of candidate parents sampled from the population
        p_mutation (float) - mutation probability
        p_crossover (float) - crossover probability
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
        
        Output
        round_log (list) - evolution round description
        '''
        round_log = []
        for rank,individual in enumerate(self.population):
            params_list = []
            for param in individual.params.keys():
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
    
    def evolution(self, opt_function, n_rounds, n_children, n_sample, p_mutation, p_crossover):
        '''
        Executes the evolution algorithm for a set number of iterations
        
        Arguments
        opt_function (function) - function to be optimised
        n_rounds (int) - number of evolution rounds
        n_children (int) - number of children
        n_sample (int) - number of candidate parents sampled from the population
        p_mutation (float) - mutation probability
        p_crossover (float) - crossover probability
        
        Ouput
        log_df (dataframe) - dataframe containing a summary of the evolution process, 
        with a row by id/stage combination
        '''
        log_list = []
        for i in range(n_rounds):
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
        
        Argument
        log_df (dataframe) - dataframe containing a summary of the evolution process
        '''
        log_df = log_df[log_df['rank']==1]
        ax = log_df.plot('stage', 'fitness') 
        ax.set_title('Fitness over time', fontsize = 20, pad=20)
        ax.set_xlabel('Generation (Program Iteration)', fontsize=12)
        ax.set_ylabel('Fitness', fontsize=12)
        
    def network_genealogy(self,log_df):
        '''
        Generates a networkX compatible genealogy in a pandas dataframe format
        
        Argument
        log_df (dataframe) - pandas dataframe containing a summary of the evolution process
        
        Output
        genealogy (dataframe) - pandas dataframe convertible to a NetworkX graph
        '''
        pass
    
    def get_population_params(self):
        '''
        Generates a list of the population's parameters
        
        Output
        param_list (list) - a list of the poulation's individuals' parameters
        '''
        param_list = []
        for ind in self.population:
            param_list.append(ind.params)
        return param_list
    
    def get_population_params_fitness(self):
        '''
        Generates a list of the population's parameters and their associated fitness
        
        Output
        param_fitness (list) - list of dictionaries containing each individual's parameters and fitness
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


# In[5]:


def optimise(function, search_space,
             minimize=True, population_size=20,
             n_rounds=500, n_children=10, 
             n_sample=4, p_mutation=0.2, 
             p_crossover=0.6, return_log = False, 
             return_population = False):
    
    '''
    Args:
        minimize (bool, default=True): sort direction
    '''
    
    pop = Population(population_size, search_space=search_space, minimize = minimize)
    pop.get_initial_population()
    pop.evaluate_population(function)
    run_log = pop.evolution(function, n_rounds, n_children, n_sample, p_mutation, p_crossover)
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
        return best_params


# In[6]:


def solve(function, target_value, search_space, population_size=10,
             n_rounds=500, n_children=20, 
             n_sample=4, p_mutation=0.2, 
             p_crossover=0.6, return_log = False, 
             return_population = False):
    
    def absolute_error(params):
        return abs(function(params)-target_value)
    
    pop = Population(population_size, search_space=search_space, minimize = True)
    pop.get_initial_population()
    pop.evaluate_population(absolute_error)
    run_log = pop.evolution(absolute_error, n_rounds, n_children, n_sample, p_mutation, p_crossover)
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
        return best_params
    


# In[ ]:





# In[ ]:




