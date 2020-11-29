#!/usr/bin/python

import csv
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


EPISODES = 'Episodes'
TIMESTEPS = 'Timesteps'
MEAN_REWARD = '100-Episode Mean Reward'
SUCCESS_RATE = 'Success Rate'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--input-dir', type=str, default='./logs')
    parser.add_argument('--output-dir', type=str, default='./plots')
    args = parser.parse_args()


    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    data_dir = args.input_dir
    # Get all the directories with environment name
    dirs = glob.glob(os.path.join(data_dir, args.env_name + '_*'))
    # This gives me a list of like:
        # cheese_final_16/
        # cheese_final_32/
        # cheese_future_64/
        # cheese_future_32/
    # Is that what I really want?
    # Fuck it, plot the anarchy Update: anarchy does not look good

    for dir_name in dirs:
        with open(os.path.join(dir_name, 'output.csv'), 'r') as csvfile:
            df = pd.read_csv(csvfile, names=[EPISODES, TIMESTEPS, MEAN_REWARD, SUCCESS_RATE],
                             header=None)

            plt.plot(df[EPISODES], df[MEAN_REWARD], label=dir_name)

            plt.legend()
    plt.savefig(os.path.join(output_dir, args.env_name + '_episodes'))
    fig = plt.figure()
    for dir_name in dirs:
        with open(os.path.join(dir_name, 'output.csv'), 'r') as csvfile:
            # reader = csv.reader(csvfile, delimiter=',')
            df = pd.read_csv(csvfile, names=[EPISODES, TIMESTEPS, MEAN_REWARD, SUCCESS_RATE],
                             header=None)
            plt.plot(df[TIMESTEPS], df[MEAN_REWARD], label=dir_name)

            plt.legend()
    plt.savefig(os.path.join(output_dir, args.env_name + '_timesteps'))


if __name__ == '__main__':
    main()