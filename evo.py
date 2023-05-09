from src.evogfuzz.evogfuzz_class import EvoGFuzz
from src.evogfuzz.fitness_functions import (
    fitness_function_time as fit_time,
)
from fuzzingbook.Parser import EarleyParser, tree_to_string, DerivationTree
from fuzzingbook.Grammars import Grammar, simple_grammar_fuzzer
import json
import time

GRAMMAR_BUGGY_FUNCTION =  {
       "<start>":
            ["<arith>"],
        "<arith>":
            ["cov(<expr>)", "mem(<expr>)", "cpu(<expr>)", "error(<expr>)", "time(<expr>)","exception(<expr>)"],
        "<expr>":
             ["<digit><digit>", "<digit>"],

        "<digit>":
            ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    }
def buggy_function(inp):
    inp = str(inp)
    bigNumber= 1
        
        #raise ValueError("Input number must be greater than 0")
    if ("mem" in inp):
        # Simulate high memory usage
        bigList = sum(list(range(10000)))
    if ("cpu" in inp):
        # Simulate high CPU usage
        i=0
        while i < 10000:
            bigNumber = bigNumber * i
            i= i+1
    if ("time" in inp):
        # Simulate high time usage
        time.sleep(0.1)
    if ("cov" in inp):
        #Simulate more lines for coverage testing
        doSomething = 1
    if ("error" in inp):
         # Simulate TypeError
        raise TypeError("TypeError been thrown")
    if ("exception" in inp):
         # Simulate ValueError
        raise ValueError("ValueError been thrown")

def main():
   
    with open('inital_inputs.txt', 'r') as f:
        INITIAL_INPUTS = json.load(f)

    for i in range(2):
        run = i + 1
        epp = EvoGFuzz(
            GRAMMAR_BUGGY_FUNCTION,
            buggy_function,
            INITIAL_INPUTS,
            fitness_function=fit_time,
            fitnessType="wctime",
            which_rerun=run,
            iterations=30,
            with_mutations=False,
            working_dir=None,
        )
        epp.fuzz()


if __name__ == "__main__":
    main()
