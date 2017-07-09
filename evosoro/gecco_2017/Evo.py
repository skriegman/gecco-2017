#!/usr/bin/python
import random
import os
import sys
import numpy as np
import subprocess as sub

sys.path.append(os.getcwd() + "/../..")

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import DirectEncoding
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import ParetoOptimization
from evosoro.tools.checkpointing import continue_from_checkpoint


sub.call("cp ../_voxcad/voxelyzeMain/voxelyze .", shell=True)

SEED = 1
MAX_TIME = 1
POP_SIZE = 5

IND_SIZE = (4, 4, 3)
MAX_GENS = 10000
NUM_RANDOM_INDS = 1

SIM_TIME = 8.5+2.5  # includes init times
INIT_TIME = 0.5
AFTERLIFE_TIME = 0
MID_LIFE_FREEZE_TIME = 2.5  # includes another init time

FALLING_PROHIBITED = True

DT_FRAC = 0.35
MIN_TEMP_FACT = 0.25
GROWTH_AMPLITUDE = 0.75
MIN_GROWTH_TIME = 0.10
TEMP_AMP = 39

MUT_RATE = 0.5

NORMALIZE_DIST_BY_VOL = True
NORMALIZATION_EXPONENT = 0.999999
TIME_BETWEEN_TRACES = 0.01  # in sec
SAVE_TRACES = False

TIME_TO_TRY_AGAIN = 12
MAX_EVAL_TIME = 30

SAVE_VXA_EVERY = 100
SAVE_LINEAGES = True
CHECKPOINT_EVERY = 1
EXTRA_GENS = 0

RUN_DIR = "run_{}".format(SEED)
RUN_NAME = "Evo"


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(DirectEncoding(output_node_name="initial_size", orig_size_xyz=IND_SIZE, scale=1, p=MUT_RATE))
        self.to_phenotype_mapping.add_map(name="initial_size", tag="<InitialVoxelSize>")


if not os.path.isfile("./" + RUN_DIR + "/pickledPops/Gen_0.pickle"):

    random.seed(SEED)
    np.random.seed(SEED)

    my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT,
                 fitness_eval_init_time=INIT_TIME, afterlife_time=AFTERLIFE_TIME,
                 mid_life_freeze_time=MID_LIFE_FREEZE_TIME)

    my_env = Env(temp_amp=TEMP_AMP, time_between_traces=TIME_BETWEEN_TRACES)
    my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")
    my_env.add_param("min_growth_time", MIN_GROWTH_TIME, "<MinGrowthTime>")
    my_env.add_param("falling_prohibited", int(FALLING_PROHIBITED), "<FallingProhibited>")

    my_env.add_param("norm_dist_by_vol", int(NORMALIZE_DIST_BY_VOL), "<NormDistByVol>")
    my_env.add_param("normalization_exponent", int(NORMALIZATION_EXPONENT), "<NormalizationExponent>")
    my_env.add_param("save_traces", int(SAVE_TRACES), "<SaveTraces>")

    my_objective_dict = ObjectiveDict()
    my_objective_dict.add_objective(name="fitness", maximize=True, tag="<NormFinalDist>")
    my_objective_dict.add_objective(name="age", maximize=False, tag=None)
    my_objective_dict.add_objective(name="frozen_dist", maximize=True, tag="<NormFrozenDist>", logging_only=True)

    my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POP_SIZE)

    my_optimization = ParetoOptimization(my_sim, my_env, my_pop)
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)

else:
    continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
                             max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
                             checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_VXA_EVERY,
                             save_lineages=SAVE_LINEAGES)
