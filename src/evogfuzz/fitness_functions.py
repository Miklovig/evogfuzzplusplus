from evogfuzz.input import Input
from evogfuzz.oracle import OracleResult
import logging
#############---------FITNESS FAILURE----------------################
def fitness_function_failure(
    test_input: Input
) -> float:
    return get_fitness(test_input)

def get_fitness(test_input: Input) -> int:
    if test_input.oracle == OracleResult.BUG:
        return 1
    else:
        return 0
    
#############----------FITNESS TIME---------------################

def fitness_function_time(
     test_input: Input
) -> float:
    fitness = get_fitness_time(test_input)
    return fitness

def get_fitness_time(test_input: Input) -> int:
    #if its a bug, then we dont run the whole app, but crash somewhere in the middle
    #so it has to be zero to elliminate those cases
    if test_input.oracle == OracleResult.BUG:
        return 0
    else:
        return test_input.exec_feature


#############----------FITNESS EXCEPTION---------------################
#for now we can test for specified error, like ValueError or NameError
#in the future we want to test whether there was an error or not
def fitness_function_except(
    test_input: Input
)-> float:
    return get_fitness_except(test_input)

def get_fitness_except(test_input: Input) -> int:
    if (test_input.error_message == ValueError):
        return 1
    else:
        return 0
    


#############----------FITNESS MEMORY---------------################

def fitness_function_memory(
    test_input: Input
)-> float:
    return get_fitness_memory(test_input)

def get_fitness_memory(test_input: Input) -> int:
    if (test_input.oracle == OracleResult.BUG):
        return 0
    else:
        return test_input.exec_feature
    
#############----------FITNESS COVERAGE---------------################

def fitness_function_coverage(
    test_input: Input
)-> float:
    return get_fitness_coverage(test_input)

def get_fitness_coverage(test_input: Input) -> int:
    logging.info(f"in coverage: {test_input.oracle}, {test_input.exec_feature}")
    if (test_input.oracle == OracleResult.BUG):
        return 0
    else:
        return test_input.exec_feature
