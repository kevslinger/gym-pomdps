import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
import os
import yaml
import pandas as pd
import glob

EPISODES = 'Episodes'
TIMESTEPS = 'Timesteps'
MEAN_REWARD = 'Mean Reward'
SUCCESS_RATE = 'Success Rate'
STATS = (EPISODES, TIMESTEPS, MEAN_REWARD, SUCCESS_RATE)


def get_config():
    with open('_figures.yaml', 'r') as f:
        return yaml.safe_load(f.read())


def create_report():
    report = dict()
    for stat in STATS:
        report[stat] = []
    return report


def parse_data(input_dir, experiment):
    print(os.path.join(input_dir, experiment + '_1'))
    # Get the list of files (we have multiple seeds, and each directory will have its own seed)
    #data_file_list = [glob.glob(os.path.join(input_dir, experiment + '_' + str(k), '*.csv'))[0] for k in range(5)]
    data_file_list = glob.glob(os.path.join(input_dir, experiment, '*.csv'))
    report = create_report()
    # Read in each file, add its data to our report, then average all the data together.
    for path in data_file_list:
        df = pd.read_csv(path, names=list(STATS),
                     header=None)
        for stat in STATS:
            report[stat].append(df[stat])
    for stat in STATS:
        data = report.pop(stat)

        mean = np.mean(data, axis=0)

        report[stat] = mean
    return report

def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./logs')
    parser.add_argument('--output_dir', type=str, default='./auto_plots')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    config = get_config()

    for plot_group, group_params in config.items():
        for plot_name, plot_params in group_params['plots'].items():
            plt.figure()
            for e, label, color in zip(plot_params['experiments'],
                                        group_params['labels'],
                                        group_params['colors']):
                plt.xlabel(group_params['xlabel'])
                plt.ylabel(group_params['ylabel'])
                plt.title(plot_params['title'])

                data = parse_data(args.input_dir, e)
                try:
                    x, y = map(np.array, [data[TIMESTEPS], data[MEAN_REWARD]])
                except:
                    print(f"Could not map f{TIMESTEPS} and f{MEAN_REWARD} for f{plot_name}")
                    continue
                if 'label' in plot_params:
                    label = plot_params['label']
                plt.plot(x, y, color=color, label=label, linewidth=2.0)
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, plot_name))


if __name__ == '__main__':
    main()
