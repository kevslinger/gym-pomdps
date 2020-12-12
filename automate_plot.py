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
    try:
        data_file_list = [glob.glob(os.path.join(input_dir, experiment + '_' + str(k), '*.csv'))[0] for k in range(5)]
    except IndexError:
        error_count = 1
        for i in reversed(range(5)):
            try:
                data_file_list = [glob.glob(os.path.join(input_dir, experiment + '_' + str(k), '*.csv'))[0] for k in range(i)]
            except IndexError:
                error_count += 1
                continue
            break
        if error_count >= 5:
            print(f"Error finding files for {experiment}")
            return None
    #data_file_list = glob.glob(os.path.join(input_dir, experiment, '*.csv'))
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

    i = 1
    for plot_group, group_params in config.items():
        for plot_name, plot_params in group_params['plots'].items():

            min_y = np.inf
            max_y = -np.inf
            for e, label, color in zip(plot_params['experiments'],
                                        group_params['labels'],
                                        group_params['colors']):

                data = parse_data(args.input_dir, e)

                try:
                    x, y1, y2 = map(np.array, [data[TIMESTEPS], data[MEAN_REWARD], data[SUCCESS_RATE]])
                    print(type(data))
                except:

                    print(f"Could not map {TIMESTEPS}/{MEAN_REWARD}/{SUCCESS_RATE} for {plot_name}")
                    continue


                if 'sparse' in e:
                    min_y = 0
                    max_y = 1
                    #max_y = max(max_y, max(data[MEAN_REWARD] + (0.05 - max(data[MEAN_REWARD]) % 0.05)))
                    #plt.ylim([0, max(data[MEAN_REWARD]) + (0.05 - max(data[MEAN_REWARD]) % 0.05)])
                else:
                    min_y = min(min_y, min(data[MEAN_REWARD] - (2 - min(data[MEAN_REWARD]) % 2)))
                    max_y = max(max_y, max(data[MEAN_REWARD] + (2 - max(data[MEAN_REWARD]) % 2)))
                    #plt.ylim([min(data[MEAN_REWARD]) - (5 - min(data[MEAN_REWARD]) % 5), ])

                plt.figure(i)
                if 'label' in plot_params:
                    label = plot_params['label']
                plt.plot(x, y1, color=color, label=label, linewidth=2.0)
                plt.figure(i+1)
                plt.plot(x, y2, color=color, label=label, linewidth=2.0)

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
            plt.title(plot_params['title'])
            plt.xlabel('Timesteps')
            plt.ylabel('100-Episode Mean Reward')
            plt.xlim(group_params['xlim'])
            plt.ylim([min_y, max_y])
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, plot_name + '_reward'))

            #if 'xlabel' in plot_params:
            #    plt.xlabel(plot_params['xlabel'])
            #else:
            #    plt.xlabel(group_params['xlabel'])


            plt.figure(i+1)
            plt.title(plot_params['title'])
            plt.xlabel('Timesteps')
            plt.ylabel('Success Rate')
            plt.xlim(group_params['xlim'])
            plt.ylim([0, 1.0])
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, plot_name + '_success'))
            i += 2


if __name__ == '__main__':
    main()
