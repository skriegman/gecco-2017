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
from evosoro.tools.algorithms import ParetoOptimization, SetMutRateOptimization
from evosoro.tools.checkpointing import continue_from_checkpoint


VOXELYZE_VERSION = '_voxcad'
# sub.call("rm ./voxelyze", shell=True)
sub.call("cp ../" + VOXELYZE_VERSION + "/voxelyzeMain/voxelyze .", shell=True)
# sub.call("chmod 755 ./voxelyze", shell=True)

SEED = 1
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
TEMP_AMP = 39  # base of 25 (default) is subtracted
FALLING_PROHIBITED = False

NUM_ENV_CYCLES = 50  # number of cycles through environment in MAX_GENS (for 100 we need to set as 1000 per 10000 maxg)
NUM_HURDLES = 20
HURDLE_HEIGHT = 1

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

        self.add_network(DirectEncoding(output_node_name="mutation_rate", orig_size_xyz=IND_SIZE,
                                        scale=1 / 48., p=0.25, symmetric=False,
                                        lower_bound=0, start_val=1 / 48., mutate_start_val=True))

        self.add_network(DirectEncoding(output_node_name="initial_size", orig_size_xyz=IND_SIZE, scale=1))
        self.to_phenotype_mapping.add_map(name="initial_size", tag="<InitialVoxelSize>")

        self.add_network(DirectEncoding(output_node_name="final_size", orig_size_xyz=IND_SIZE, scale=1))
        self.to_phenotype_mapping.add_map(name="final_size", tag="<FinalVoxelSize>")

        self.add_network(DirectEncoding(output_node_name="init_phase_offset", orig_size_xyz=IND_SIZE, scale=1,
                                        symmetric=False))
        self.to_phenotype_mapping.add_map(name="init_phase_offset", tag="<PhaseOffset>")

        self.add_network(DirectEncoding(output_node_name="final_phase_offset", orig_size_xyz=IND_SIZE, scale=1,
                                        symmetric=False))
        self.to_phenotype_mapping.add_map(name="final_phase_offset", tag="<FinalPhaseOffset>")

        # self.add_network(DirectEncoding(output_node_name="actuation_damping", orig_size_xyz=IND_SIZE,
        #                                 scale=1, p=1/48., lower_bound=0, upper_bound=10))
        # self.to_phenotype_mapping.add_map(name="actuation_damping", tag="<TempAmpDamp>")
        #
        # self.add_network(DirectEncoding(output_node_name="final_actuation_damping", orig_size_xyz=IND_SIZE,
        #                                 scale=1, p=1/48., lower_bound=0, upper_bound=10))
        # self.to_phenotype_mapping.add_map(name="final_actuation_damping", tag="<FinalTempAmpDamp>")


# set simulation
my_sim = Sim(dt_frac=DT_FRAC, simulation_time=SIM_TIME, min_temp_fact=MIN_TEMP_FACT, fitness_eval_init_time=INIT_TIME,
             afterlife_time=AFTERLIFE_TIME, mid_life_freeze_time=MID_LIFE_FREEZE_TIME)

# set environment
env1 = Env(temp_amp=TEMP_AMP)
env2 = Env(temp_amp=TEMP_AMP, num_hurdles=NUM_HURDLES, hurdle_height=HURDLE_HEIGHT)
my_env = [env1]  # [env1, env2]
for e in my_env:
    e.add_param("growth_amplitude", GROWTH_AMPLITUDE, "<GrowthAmplitude>")
    e.add_param("min_growth_time", MIN_GROWTH_TIME, "<MinGrowthTime>")
    e.add_param("falling_prohibited", int(FALLING_PROHIBITED), "<FallingProhibited>")

# set objectives for optimization
my_objective_dict = ObjectiveDict()
my_objective_dict.add_objective(name="fitness", maximize=True, tag="<finalDistY>")  # in positive y
my_objective_dict.add_objective(name="age", maximize=False, tag=None)

# initialize a pop of SoftBots
my_pop = Population(my_objective_dict, MyGenotype, Phenotype, pop_size=POPSIZE)

# set optimization procedure
my_optimization = SetMutRateOptimization(my_sim, my_env, my_pop, [1.0, 0.25, 0.25, 0.25, 0.25])

if __name__ == "__main__":
    my_optimization.run(max_hours_runtime=MAX_TIME, max_gens=MAX_GENS, num_random_individuals=NUM_RANDOM_INDS,
                        num_env_cycles=NUM_ENV_CYCLES,
                        directory=RUN_DIR, name=RUN_NAME, max_eval_time=MAX_EVAL_TIME,
                        time_to_try_again=TIME_TO_TRY_AGAIN, checkpoint_every=CHECKPOINT_EVERY,
                        save_vxa_every=SAVE_POPULATION_EVERY, save_lineages=SAVE_LINEAGES)

    # continue_from_checkpoint(directory=RUN_DIR, additional_gens=EXTRA_GENS, max_hours_runtime=MAX_TIME,
    #                          max_eval_time=MAX_EVAL_TIME, time_to_try_again=TIME_TO_TRY_AGAIN,
    #                          checkpoint_every=CHECKPOINT_EVERY, save_vxa_every=SAVE_POPULATION_EVERY,
    #                          save_lineages=SAVE_LINEAGES)
