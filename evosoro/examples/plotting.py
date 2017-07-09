from glob import glob
import pandas as pd
from evosoro.tools.data_analysis import get_all_data, combine_experiments, plot_time_series


EXP_NAMES = ["evo", "evodevo_2nets", "evodevo_3nets_ts", "evodevo_3nets_tf"]
SAVE_LOCATION = "best_results.csv"


exp_data = []
for exp_name in EXP_NAMES:
    files = glob('{}/run*/bestSoFar/bestOfGen.txt'.format(exp_name))
    print exp_name, len(files)
    exp_data += [get_all_data(files)]

data = combine_experiments(exp_data, EXP_NAMES)
data.to_csv(SAVE_LOCATION)

# to load previously saved data use:
# data = pd.read_csv(SAVE_LOCATION)

plot_time_series(data, "best_results.pdf")
