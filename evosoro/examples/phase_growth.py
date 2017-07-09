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
from evosoro.tools.utils import positive_sigmoid, count_negative, count_positive, rescaled_positive_sigmoid


VOXELYZE_VERSION = '_voxcad'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)
# sub.call("chmod 755 ./voxelyze", shell=True)

SEED = 1
MAX_TIME = 0.5  # in hours
POPSIZE = 30

IND_SIZE = (5, 3, 6)
MAX_GENS = 100
NUM_RANDOM_INDS = 1

BIPED = True
BIPED_LEG_PROPORTION = 4/6.
FALLING_PROHIBITED = True
MUSCLE_STIFFNESS = 5e+006

SIM_TIME = 10.5  # includes init time
INIT_TIME = 0.5

DT_FRAC = 0.9
MIN_GROWTH_TIME = 0
TEMP_AMP = 39

TIME_TO_TRY_AGAIN = 8
MAX_EVAL_TIME = 30

SAVE_VXA_EVERY = 100
SAVE_LINEAGES = True
CHECKPOINT_EVERY = 1
EXTRA_GENS = 0

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


def growth_onset(x, x_min=0.0, x_max=0.1):
    return rescaled_positive_sigmoid(x, x_min=x_min, x_max=x_max)


def frequency_func(x):
    return np.mean(rescaled_positive_sigmoid(x, x_min=2, x_max=8))


class MyGenotype(Genotype):
    def __init__(self):
        Genotype.__init__(self, orig_size_xyz=IND_SIZE)

        # self.add_network(CPPN(output_node_names=["frequency"]))
        # self.to_phenotype_mapping.add_map(name="frequency", tag="<TempPeriod>", env_kws={"frequency": frequency_func})
        #  tag actually doesn't do anything here

        self.add_network(CPPN(output_node_names=["start_growth_time"]))
        self.to_phenotype_mapping.add_map(name="start_growth_time", tag="<StartGrowthTime>", func=positive_sigmoid,
                                          logging_stats=[np.median, np.mean, np.std])

        self.add_network(CPPN(output_node_names=["growth_time"]))
        self.to_phenotype_mapping.add_map(name="growth_time", tag="<GrowthTime>", func=positive_sigmoid,
                                          logging_stats=[np.median, np.mean, np.std])

        self.add_network(CPPN(output_node_names=["initial_phase_offset"]))
        self.to_phenotype_mapping.add_map(name="initial_phase_offset", tag="<PhaseOffset>",
                                          logging_stats=[np.median, np.mean, np.std, count_negative, count_positive])

        self.add_network(CPPN(output_node_names=["final_phase_offset"]))
        self.to_phenotype_mapping.add_map(name="final_phase_offset", tag="<FinalPhaseOffset>",
                                          logging_stats=[np.median, np.mean, np.std, count_negative, count_positive])

my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, fitness_eval_init_time=INIT_TIME)

my_env = Env(temp_amp=TEMP_AMP, biped=BIPED, biped_leg_proportion=BIPED_LEG_PROPORTION,
             muscle_stiffness=MUSCLE_STIFFNESS)
my_env.add_param("min_growth_time", MIN_GROWTH_TIME, "<MinGrowthTime>")
my_env.add_param("falling_prohibited", int(FALLING_PROHIBITED), "<FallingProhibited>")

my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<PosteriorY>")
my_objective_dict.add_objective(name="age", maximize=False, tag=None)
# my_objective_dict.add_objective(name="lifetime", maximize=True, tag="<Lifetime>", logging_only=True)

my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POPSIZE)

my_optimization = ParetoOptimization(my_sim, my_env, my_pop)

if __name__ == "__main__":
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_VXA_EVERY, save_lineages=SAVE_LINEAGES)
