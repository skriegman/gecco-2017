#!/usr/bin/python
"""

In this example robots are allowed to develop as well as evolve. Robots can add and remove matter during their
lifetimes through linear volumetric changes.

The Voxel Picture
-----------------
                             Growing Case:                                  Shrinking Case:

                 vol                                             vol
                  |                                               |
    final volume _|                    ____________              _|____________
                  |                  /                            |            \
                  |                 /                             |             \
                  |                /                              |              \
                  |               /                               |               \
                  |              /                                |                \
                  |             /                                 |                 \
                  |            /                                  |                  \
    start volume _|___________/                                  _|                   \____________
                  |                                               |
                  |                                               |
                 _|___________|_______|_____________             _|___________|_______|____________
                  0           onset   offset        time          0           onset   offset       time

Additional References
---------------------

In-preparation


"""
import random
import subprocess as sub
import numpy as np
import os
import sys

# Appending repo's root dir in the python path to enable subsequent imports
sys.path.append(os.getcwd()+"/../..")

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import CPPN
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import ParetoOptimization
from evosoro.tools.utils import positive_sigmoid, mean_abs, std_abs, count_negative, count_positive
from evosoro.tools.checkpointing import continue_from_checkpoint


VOXELYZE_VERSION = '_voxcad'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)
# sub.call("chmod 755 ./voxelyze", shell=True)

SEED = 1
MAX_TIME = 0.5  # in hours
POPSIZE = 30
MAX_GENS = 100
NUM_RANDOM_INDS = 1
SIM_TIME = 10.5  # includes init time
INIT_TIME = 0.5

NUM_HURDLES = 0
SPACE_BETWEEN_HURDLES = 10
CIRCULAR_HURDLES = False
FORWARD_HURDLES_ONLY = True
FENCE = False
WALL_HEIGHT = 1
BACK_STOP = True

# new stuff for biped
BIPED = True
BIPED_LEG_PROPORTION = 3/5.
FALLING_PROHIBITED = True

IND_SIZE = (5, 3, 5)
DT_FRAC = 0.35
MIN_TEMP_FACT = 0.25  # (0.25 for 555, at least 0.30 for 777)
GROWTH_AMPLITUDE = 0.75  # (0.75 for 555, at least 0.70 for 777)
MIN_GROWTH_TIME = 0.10
TEMP_AMP = 39

TIME_BETWEEN_TRACES = 0  # in sec
NORMALIZE_DIST_BY_VOL = False
SAVE_TRACES = False  # if true, set time between traces > 0

TIME_TO_TRY_AGAIN = 10  # seconds per individual
MAX_EVAL_TIME = 60  # seconds per individual

SAVE_POPULATION_EVERY = 10
SAVE_LINEAGES = True
CHECKPOINT_EVERY = 1
EXTRA_GENS = 0  # for adding extra gens after checkpointing

RUN_DIR = "growth_data"
RUN_NAME = "Growth"


random.seed(SEED)
np.random.seed(SEED)


def randomly(x):
    return np.random.random(size=x.shape)


def randomly_neg_pos(x):
    return np.random.random(size=x.shape) * np.random.choice([-1, 1], size=x.shape)


def grow_some_min_sizes(x, p=.4):
    x[np.random.random(size=x.shape) <= p] = -1
    x[np.random.random(size=x.shape) > p] = 1
    return x


def grow_max_size(x):
    return np.ones_like(x)


def grow_min_size(x):
    return grow_max_size(x) * -1.0


def grow_immediately(x):
    return np.zeros_like(x)


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        self.add_network(CPPN(output_node_names=["initial_size"]))
        self.to_phenotype_mapping.add_map(name="initial_size", tag="<InitialVoxelSize>",
                                          logging_stats=[np.median, np.mean, np.std, count_negative, count_positive])

        self.add_network(CPPN(output_node_names=["final_size"]))
        self.to_phenotype_mapping.add_map(name="final_size", tag="<FinalVoxelSize>",
                                          logging_stats=[np.median, np.mean, np.std, count_negative, count_positive])

        self.add_network(CPPN(output_node_names=["start_growth_time"]))
        self.to_phenotype_mapping.add_map(name="start_growth_time", tag="<StartGrowthTime>", func=positive_sigmoid,
                                          logging_stats=[np.median, np.mean, mean_abs, np.std, std_abs])

        self.add_network(CPPN(output_node_names=["growth_time"]))
        self.to_phenotype_mapping.add_map(name="growth_time", tag="<GrowthTime>", func=positive_sigmoid,
                                          logging_stats=[np.median, np.mean, mean_abs, np.std, std_abs])

        # self.add_network(CPPN(output_node_names=["phase_offset"]))
        # self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>",
        #                                   logging_stats=[np.median, np.mean, mean_abs, np.std, std_abs])


# set simulation
my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT, fitness_eval_init_time=INIT_TIME)

# set environment
my_env = Env(temp_amp=TEMP_AMP, time_between_traces=TIME_BETWEEN_TRACES, num_hurdles=NUM_HURDLES,
             space_between_hurdles=SPACE_BETWEEN_HURDLES, circular_hurdles=CIRCULAR_HURDLES,
             forward_hurdles_only=FORWARD_HURDLES_ONLY, wall_height=WALL_HEIGHT, back_stop=BACK_STOP, fence=FENCE,
             biped=BIPED, biped_leg_proportion=BIPED_LEG_PROPORTION)

my_env.add_param("time_between_volume_trace", TIME_BETWEEN_TRACES, "<TimeBetweenVolumeTrace>")
my_env.add_param("norm_dist_by_vol", int(NORMALIZE_DIST_BY_VOL), "<NormDistByVol>")
my_env.add_param("save_traces", int(SAVE_TRACES), "<SaveTraces>")
my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")
my_env.add_param("min_growth_time", MIN_GROWTH_TIME, "<MinGrowthTime>")
my_env.add_param("falling_prohibited", int(FALLING_PROHIBITED), "<FallingProhibited>")

# set objectives for optimization
my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<FallAdjustedY>")
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

# initialize a pop of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POPSIZE)

# set optimization procedure
my_optimization = ParetoOptimization(my_sim, my_env, my_pop)

if __name__ == "__main__":
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)

    # Here is how to use the checkpointing mechanism
    # if not os.path.isfile("./" + RUN_DIR + "/checkpoint.pickle"):
    #     # start optimization
    #     my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
    #                         directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
    #                         time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
    #                         save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)
    #
    # else:
    #     continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
    #                              max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
    #                              checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_POPULATION_EVERY,
    #                              save_lineages=SAVE_LINEAGES)
