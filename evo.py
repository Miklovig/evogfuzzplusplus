from src.evogfuzz.evogfuzz_class import EvoGFuzz
from src.evogfuzz.fitness_functions import (
    fitness_function_time as fit_time,
)
from fuzzingbook.Parser import EarleyParser, tree_to_string, DerivationTree
from fuzzingbook.Grammars import Grammar, simple_grammar_fuzzer
from fuzzingbook.GeneratorGrammarFuzzer import ProbabilisticGeneratorGrammarFuzzer
import json
import time
import subprocess
import logging
from importlib import reload  # Not needed in Python 2

GRAMMAR_BUGGY_FUNCTION3 : Grammar =  {
       "<start>":
            ["<arith>"], 
        "<arith>":
            ["cov(<expr>)", "mem(<expr>)", "cpu(<expr>)", "error(<expr>)", "time(<expr>)","exception(<expr>)"],
        "<expr>":
             ["<digit><digit>", "<digit>"],

        "<digit>":
            ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

    }
def sub_buggy_c(inp):

    # Compile the C program
    # Execute the compiled C program with specified inputs
    result = subprocess.run(["./buggy_evo", str(inp)],  capture_output=True)
    logging.info(result.stdout)


def main():
   
    INITIAL_INPUTS = ["time(5)", "time(8)", "exception(3)", "cov(6)", "time(3)", "time(68)", "time(22)", "cpu(2)", "mem(2)", "cpu(13)", "exception(7)", "cov(15)", "exception(34)", "exception(92)"]


    for i in range(2):
        run = i + 1
        epp = EvoGFuzz(
            GRAMMAR_BUGGY_FUNCTION3,
            sub_buggy_c,
            INITIAL_INPUTS,
            fitness_function=fit_time,
            fitnessType="cputime",
            which_rerun=run,
            iterations=30000,
            with_mutations=True,
            working_dir=None,
            )
        epp.fuzz()
        print(f"Done with the {i}. run")


if __name__ == "__main__":
    main()
