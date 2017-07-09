import random
import subprocess as sub
import numpy as np
import os
import sys

# Appending repo's root dir in the python path to enable subsequent imports
sys.path.append(os.getcwd()+"/../..")

from evosoro.base import Sim, Env, ObjectiveDict
from evosoro.networks import CPPN, DirectEncoding
from evosoro.softbot import Genotype, Phenotype, Population
from evosoro.tools.algorithms import ParetoOptimization, GenomeWideMutationOptimization
from evosoro.tools.checkpointing import continue_from_checkpoint
from evosoro.tools.utils import positive_sigmoid, make_material_tree, count_occurrences, \
    mean_abs, std_abs, count_negative, count_positive, rescaled_positive_sigmoid


VOXELYZE_VERSION = '_voxcad'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)
# sub.call("chmod 755 ./voxelyze", shell=True)

SEED = 10
MAX_TIME = 0.5  # in hours
POPSIZE = 5

IND_SIZE = (4, 4, 3)
MAX_GENS = 100
NUM_RANDOM_INDS = 1

SIM_TIME = 2.5  # includes init time
INIT_TIME = 0.5
AFTERLIFE_TIME = 0
MID_LIFE_FREEZE_TIME = 0

DT_FRAC = 0.35
MIN_TEMP_FACT = 0.25
GROWTH_AMPLITUDE = 0.75
MIN_GROWTH_TIME = 0.10
TEMP_AMP = 39
FALLING_PROHIBITED = False

NUM_HURDLES = 0
HURDLE_HEIGHT = -1
SPACE_BETWEEN_HURDLES = 3
HURDLE_STOP = 20
DEBRIS = False
DEBRIS_SIZE = 0
DEBRIS_START = -2
TUNNEL_WIDTH = 8
SQUEEZE_RATE = 0
SQUEEZE_START = 1
SQUEEZE_END = 3
CONSTANT_SQUEEZE = False
CIRCULAR_HURDLES = False
FORWARD_HURDLES_ONLY = True
BACK_STOP = False
FENCE = False
WALL_HEIGHT = 3

NORMALIZE_DIST_BY_VOL = False
TIME_BETWEEN_TRACES = 0.01  # in sec
SAVE_TRACES = False
NORMALIZATION_EXPONENT = 0.99999

TIME_TO_TRY_AGAIN = 20  # seconds per individual
MAX_EVAL_TIME = 60  # seconds per individual

SAVE_POPULATION_EVERY = 100
SAVE_LINEAGES = True
PICKLED_POPS = True
CHECKPOINT_EVERY = 1
EXTRA_GENS = 0  # for adding extra gens after checkpointing

RUN_DIR = "growth_data"
RUN_NAME = "Growth"


random.seed(SEED)
np.random.seed(SEED)


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # self.add_network(CPPN(output_node_names=["initial_size", "final_size"]))
        self.add_network(DirectEncoding(output_node_name="initial_size", orig_size_xyz=IND_SIZE, scale=1, p=1.))
        self.to_phenotype_mapping.add_map(name="initial_size", tag="<InitialVoxelSize>")

        self.add_network(DirectEncoding(output_node_name="final_size", orig_size_xyz=IND_SIZE, scale=1, p=1.))
        self.to_phenotype_mapping.add_map(name="final_size", tag="<FinalVoxelSize>")

        # self.add_network(CPPN(output_node_names=["phase_offset"]))
        self.add_network(DirectEncoding(output_node_name="phase_offset", orig_size_xyz=IND_SIZE, scale=1, p=1.))
        self.to_phenotype_mapping.add_map(name="phase_offset", tag="<PhaseOffset>")
        #
        self.add_network(DirectEncoding(output_node_name="final_phase_offset", orig_size_xyz=IND_SIZE, scale=1, p=1.))
        self.to_phenotype_mapping.add_map(name="final_phase_offset", tag="<FinalPhaseOffset>")

        # self.add_network(CPPN(output_node_names=["shape"]))
        #
        # self.to_phenotype_mapping.add_map(name="material", tag="<Data>", func=make_material_tree,
        #                                   dependency_order=["shape"], output_type=int)
        #
        # self.to_phenotype_mapping.add_output_dependency(name="shape", dependency_name=None, requirement=None,
        #                                                 material_if_true="4", material_if_false="0")


# Define a custom phenotype, inheriting from the Phenotype class
class MyPhenotype(Phenotype):
    def is_valid(self, min_percent_full=0.3, min_percent_muscle=0.1):
        # override super class function to redefine what constitutes a valid individuals
        for name, details in self.genotype.to_phenotype_mapping.items():
            if np.isnan(details["state"]).any():
                return False
            if name == "material":
                state = details["state"]
                # Discarding the robot if it doesn't have at least a given percentage of non-empty voxels
                if np.sum(state > 0) < np.product(self.genotype.orig_size_xyz) * min_percent_full:
                    return False
                # Discarding the robot if it doesn't have at least a given percentage of muscles (materials 3 and 4)
                if count_occurrences(state, [3, 4]) < np.product(self.genotype.orig_size_xyz) * min_percent_muscle:
                    return False
        return True

# set simulation
my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT, fitness_eval_init_time=INIT_TIME,
             afterlife_time=AFTERLIFE_TIME, mid_life_freeze_time=MID_LIFE_FREEZE_TIME)

# set environment
my_env = Env(temp_amp=TEMP_AMP, time_between_traces=TIME_BETWEEN_TRACES,
             num_hurdles=NUM_HURDLES,
             hurdle_height=HURDLE_HEIGHT, space_between_hurdles=SPACE_BETWEEN_HURDLES, hurdle_stop=HURDLE_STOP,
             debris=DEBRIS, debris_size=DEBRIS_SIZE, debris_start=DEBRIS_START, squeeze_rate=SQUEEZE_RATE,
             squeeze_start=SQUEEZE_START, squeeze_end=SQUEEZE_END, constant_squeeze=CONSTANT_SQUEEZE,
             circular_hurdles=CIRCULAR_HURDLES, forward_hurdles_only=FORWARD_HURDLES_ONLY,
             tunnel_width=TUNNEL_WIDTH, back_stop=BACK_STOP, fence=FENCE, wall_height=WALL_HEIGHT
             )

my_env.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")
my_env.add_param("min_growth_time", MIN_GROWTH_TIME, "<MinGrowthTime>")
my_env.add_param("falling_prohibited", int(FALLING_PROHIBITED), "<FallingProhibited>")

my_env.add_param("norm_dist_by_vol", int(NORMALIZE_DIST_BY_VOL), "<NormDistByVol>")
my_env.add_param("normalization_exponent", int(NORMALIZATION_EXPONENT), "<NormalizationExponent>")
my_env.add_param("save_traces", int(SAVE_TRACES), "<SaveTraces>")


# set objectives for optimization
my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<finalDistY>")  # in positive y
my_objective_dict.add_objective(name="age", maximize=False, tag=None)
# my_objective_dict.add_objective(name="frozen_dist", maximize=True, tag="<NormFrozenDist>", logging_only=True)
# my_objective_dict.add_objective(name="regime_dist", maximize=True, tag="<NormRegimeDist>", logging_only=True)

# initialize a pop of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, MyPhenotype, pop_size=POPSIZE)

# set optimization procedure
my_optimization = GenomeWideMutationOptimization(my_sim, my_env, my_pop)

if __name__ == "__main__":
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)

    # continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
    #                          max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
    #                          checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_POPULATION_EVERY,
    #                          save_lineages=SAVE_LINEAGES)
