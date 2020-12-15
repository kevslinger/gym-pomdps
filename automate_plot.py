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

# These graphs are gonna be busyyyyyyyy
COLORS = ('tab:cyan', 'tab:red', 'tab:purple', 'tab:blue', 'tab:orange', 'tab:green')
#ENV_NAMES = ('cheese', 'cheeseonehot', 'hallway', 'hallwayonehot', 'mit', 'mitonehot', 'cit', 'citonehot')
ENV_NAMES = ('cheese', 'cheeseonehot', 'hallway', 'hallwayonehot')
GOAL_SELECTION_STRATEGIES = ('final', 'future')
REWARD_TYPES = ('dense', 'sparse')
LAYER_SIZES = (16, 32, 64)
#LAYER_SIZES = (16, 32)
STEP_CAPS = (10, 15, 20, 50, 100, np.inf)
LABELS = ('10', '15', '20', '50', '100', 'np.inf')


def get_config():
    with open('figures.yaml', 'r') as f:
        return yaml.safe_load(f.read())


def create_report():
    report = dict()
    for stat in STATS:
        report[stat] = []
    return report


def parse_data(input_dir, experiment):
    # Get the list of files (we have multiple seeds, and each directory will have its own seed)
    try:
        data_file_list = [glob.glob(os.path.join(input_dir, experiment + '_*', '*.csv'))[0]]
    except IndexError:
        print(f"Error finding files for {experiment}")
        return None
    #data_file_list = glob.glob(os.path.join(input_dir, experiment, '*.csv'))
    report = create_report()
    # Read in each file, add its data to our report, then average all the data together.
    for path in data_file_list:
        df = pd.read_csv(path, names=list(STATS), header=None)
        for stat in STATS:
            report[stat].append(df[stat])
    for stat in STATS:
        data = report.pop(stat)

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1)
        
        report[stat] = mean
        report[stat + '.std'] = std
    return report

def main():
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default='./logs')
    parser.add_argument('--output-dir', type=str, default='./auto_plots')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    #config = get_config()

    i = 1
    for env in ENV_NAMES:
        for goal_selection_strategy in GOAL_SELECTION_STRATEGIES:
            for reward_type in REWARD_TYPES:
                for layer_size in LAYER_SIZES:
                    input_dir = args.input_dir + '_' + str(layer_size)
                    output_dir = args.output_dir + '_' + str(layer_size)
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    # Make new figures here, plot all lines from step caps onto that figure
                    plot_name = '_'.join((env, goal_selection_strategy, reward_type))
                    min_y = np.inf
                    max_y = -np.inf
                    for idx, step_cap in enumerate(STEP_CAPS):
    #for plot_group, group_params in config.items():
    #    for plot_name, plot_params in group_params['plots'].items():

                       
            
            #for e, label, color in zip(plot_params['experiments'],
            #                            group_params['labels'],
            #                            group_params['colors']):

                        data = parse_data(input_dir, '_'.join((plot_name, LABELS[idx])))

                        try:
                            x, y1, y2 = map(np.array, [data[TIMESTEPS], data[MEAN_REWARD], data[SUCCESS_RATE]])
                        except TypeError:

                            print(f"Could not map {TIMESTEPS}/{MEAN_REWARD}/{SUCCESS_RATE} for {plot_name}")
                            continue

                        if 'sparse' in reward_type:
                            min_y = 0
                            max_y = 1
                            #max_y = max(max_y, max(data[MEAN_REWARD] + (0.05 - max(data[MEAN_REWARD]) % 0.05)))
                            #plt.ylim([0, max(data[MEAN_REWARD]) + (0.05 - max(data[MEAN_REWARD]) % 0.05)])
                        else:
                            min_y = min(min_y, min(data[MEAN_REWARD] - (2 - min(data[MEAN_REWARD]) % 2)))
                            max_y = max(max_y, max(data[MEAN_REWARD] + (2 - max(data[MEAN_REWARD]) % 2)))
                            #plt.ylim([min(data[MEAN_REWARD]) - (5 - min(data[MEAN_REWARD]) % 5), ])

                        plt.figure(i)
                        #if 'label' in plot_params:
                        label = LABELS[idx]
                        stddev = np.asarray(data[MEAN_REWARD + '.std'])
                        stderr = stddev / np.sqrt(5) #TODO: UPDATE AS NUMBER OF TRIALS/SEEDS GOES UP
                        plt.plot(x, y1, color=COLORS[idx], label=label, linewidth=2.0)
                        #plt.fill_between(x, (y1 - stderr), (y1 + stderr), color=COLORS[idx], alpha=0.4, linewidth=0)
                        stddev = np.asarray(data[SUCCESS_RATE + '.std'])
                        #print(stddev)
                        stderr = stddev / np.sqrt(5) # TODO: UPDATE AS NUMBER OF TRIALS/SEEDS GOES UP
                        #print(stderr)
                        plt.figure(i+1)
                        plt.plot(x, y2, color=COLORS[idx], label=label, linewidth=2.0)
                        #plt.fill_between(x, (y2 - stderr), (y2 + stderr), color=COLORS[idx], alpha=0.4, linewidth=0)

            #if 'xlabel' in plot_params:
            #    plt.xlabel(plot_params['xlabel'])
            #else:
            #    plt.xlabel(group_params['xlabel'])
            #if 'ylabel' in plot_params:
            #    plt.ylabel(plot_params['ylabel'])
            #else:
            #    plt.ylabel(group_params['ylabel'])

                    if max_y == -np.inf or min_y == np.inf:
                        continue
                    plt.figure(i)
                    #plt.title(plot_params['title'])
                    plt.title(plot_name)
                    plt.xlabel('Timesteps')
                    plt.ylabel('100-Episode Mean Reward')
                    #plt.xlim(group_params['xlim'])
                    plt.xlim([0, 200000])
                    plt.ylim([min_y, max_y])
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, plot_name + '_reward'))

            #if 'xlabel' in plot_params:
            #    plt.xlabel(plot_params['xlabel'])
            #else:
            #    plt.xlabel(group_params['xlabel'])

                    plt.figure(i+1)
                    #plt.title(plot_params['title'])
                    plt.title(plot_name)
                    plt.xlabel('Timesteps')
                    plt.ylabel('Success Rate')
                    #plt.xlim(group_params['xlim'])
                    plt.xlim([0, 200000])
                    plt.ylim([0, 1.0])
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, plot_name + '_success'))
                    i += 2


if __name__ == '__main__':
    main()
