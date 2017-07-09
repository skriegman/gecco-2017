"""

Only grow and shrink voxels of a fixed topology (material 3).

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
from evosoro.tools.utils import positive_sigmoid, mean_abs, std_abs, count_negative, \
    count_positive, make_material_tree, count_occurrences, discretize_material


VOXELYZE_VERSION = '_voxcad'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)
# sub.call("chmod 755 ./voxelyze", shell=True)

NUM_RANDOM_INDS = 1
MAX_GENS = 1000
POPSIZE = 10
IND_SIZE = (5, 5, 4)
SIM_TIME = 10
INIT_TIME = 0.4
DT_FRAC = 0.4
GROWTH_AMPLITUDE = 0.75  # if this is 7  (previously limited to .5)
MIN_TEMP_FACT = 1e-6  # and this is 2 then instability without actuation (previously limit was 0.4)
SAVE_VXA_EVERY = 10
TIME_TO_TRY_AGAIN = 10
MAX_EVAL_TIME = 60

SEED = 1
random.seed(SEED)
np.random.seed(SEED)


def grow_instantly(x):
    return np.zeros_like(x)


def grow_half_time(x):
    return np.ones_like(x) * 0.5


def grow_full_time(x):
    return np.ones_like(x)


def randomly(x):
    return np.random.random(size=x.shape)


def randomly_neg_pos(x):
    return np.random.random(size=x.shape) * np.random.choice([-1, 1], size=x.shape)


def grow_max_size(x):
    return np.ones_like(x)


def grow_min_size(x):
    return grow_max_size(x) * -1.0


def two_muscles(output_state):
    bins = np.linspace(-1, 1, num=2)
    return np.digitize(output_state, bins) + 2


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)
        # self.add_network(CPPN(output_node_names=["material"]))
        # self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=two_muscles, output_type=int)

        # self.add_network(CPPN(output_node_names=["initial_size"]))
        # self.to_phenotype_mapping.add_map(name="initial_size", tag="<InitialVoxelSize>",  func=grow_max_size,
        #                                   logging_stats=[mean_abs, std_abs, count_negative, count_positive])

        # self.add_network(CPPN(output_node_names=["start_growth_time"]))
        # self.to_phenotype_mapping.add_map(name="start_growth_time", tag="<StartGrowthTime>", func=positive_sigmoid,
        #                                   logging_stats=[mean_abs, std_abs])

        self.add_network(CPPN(output_node_names=["growth_time"]))
        self.to_phenotype_mapping.add_map(name="growth_time", tag="<GrowthTime>", func=positive_sigmoid,
                                          logging_stats=[mean_abs, std_abs])

        self.add_network(CPPN(output_node_names=["final_size"]))
        self.to_phenotype_mapping.add_map(name="final_size", tag="<FinalVoxelSize>",
                                          logging_stats=[mean_abs, std_abs, count_negative, count_positive])

        # self.add_network(CPPN(output_node_names=["shape"]))
        # self.add_network(CPPN(output_node_names=["muscleOrTissue", "muscleType", "tissueType"]))
        #
        # self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
        #                                   dependency_order=["shape", "muscleOrTissue", "muscleType"], output_type=int)
        #
        # self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
        #                                                 material_if_true=None, material_if_false="0")
        #
        # self.to_phenotype_mapping.add_output_dependency(name="muscleOrTissue", dependency_name="shape",
        #                                                 requirement=True, material_if_true=None, material_if_false="1")
        #
        # self.to_phenotype_mapping.add_output_dependency(name="tissueType", dependency_name="muscleOrTissue",
        #                                                 requirement=False, material_if_true="1", material_if_false="2")
        #
        # self.to_phenotype_mapping.add_output_dependency(name="muscleType", dependency_name="muscleOrTissue",
        #                                                 requirement=True, material_if_true="3", material_if_false="4")


# class MyPhenotype(Phenotype):
#     def is_valid(self, min_percent_full=0.1, min_percent_muscle=0.1):
#         # override super class function to redefine what constitutes a valid individuals
#         for name, details in self.genotype.to_phenotype_mapping.items():
#             if np.isnan(details["state"]).any():
#                 return False
#             if name == "material":
#                 state = details["state"]
#                 # Discarding the robot if it doesn't have at least a given percentage of non-empty voxels
#                 if np.sum(state > 0) < np.product(self.genotype.orig_size_xyz) * min_percent_full:
#                     return False
#                 # Discarding the robot if it doesn't have at least a given percentage of muscles (materials 3 and 4)
#                 if count_occurrences(state, [3, 4]) < np.product(self.genotype.orig_size_xyz) * min_percent_muscle:
#                     return False
#         return True


# set simulation
my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT, fitness_eval_init_time=INIT_TIME)

# set environment
my_env = Env()
my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")

# set objectives for optimization
my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<NormFinalDist>")
my_objective_dict.add_objective(name="age", maximize=False, tag=None)
# my_objective_dict.add_objective(name="num_voxels", maximize=False, tag=None,
#                                 node_func=np.count_nonzero, output_node_name="material")

# initialize a pop of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POPSIZE)

# set optimization procedure
my_optimization = ParetoOptimization(my_sim, my_env, my_pop)

if __name__ == "__main__":
    my_optimization.run(max_gens=MAX_GENS, save_vxa_every=SAVE_VXA_EVERY, num_random_individuals=NUM_RANDOM_INDS,
                        time_to_try_again=TIME_TO_TRY_AGAIN, max_eval_time=MAX_EVAL_TIME)
