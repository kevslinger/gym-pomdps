from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import inspect
import multiprocessing
import os
from subprocess import Popen, check_output
import threading

import yaml


T = 1_000_000
# Change cap size and layer size and goal_selection_strategy  in yaml file. Keep log dir here.
ENV_DEFAULTS = {
    'cheese': '--logdir ./logs/cheese',
    'cheeseonehot': '--logdir ./logs/cheeseonehot',
    'hallway': '--logdir ./logs/hallway',
    'hallwayonehot': '--logdir ./logs/hallwayonehot',
    'mit': '--logdir ./logs/mit',
    'mitonehot': '--logdir ./logs/mitonehot',
    'cit': '--logdir ./logs/citonehot',
}


class Dispatcher:
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
      
        self.call_script, args = list(config.items())[0]
        self.call_script = self.call_script.split(' ')
        jobs = cartesian_product(args)

        self.logdir_list = tuple(self._logdir(j) for j in jobs)
        self.cmd_list = tuple(self._cmd(j, ld) for j, ld in zip(jobs, self.logdir_list))

        self.lock = threading.Lock()
      
    @property
    def max_parallel(self):
        return multiprocessing.cpu_count()

    def execute(self):
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            for cmd, logdir in zip(self.cmd_list, self.logdir_list):
                executor.submit(self._run, cmd, logdir)


    def _cmd(self, job, logdir):
        cmd = [*self.call_script]

        # First get all of our args from the yaml config
        for key, value in job:
            cmd.extend(['--' + key, value])

        # Now add our algorithm defaults
        #env = self._env(job)
        #defaults = ENV_DEFAULTS.get(env, '')
        #cmd.extend(defaults.split(' '))

        # Finally, add the logdir
        cmd.extend(['--logdir', logdir])
        return cmd

    def _logdir(self, job):
        filename = []
        for key, value in job:
            filename.append(value.replace('_', ''))
        filename = '_'.join(filename)
        return os.path.join(self.output_dir, filename)

    def _env(self, job):
        for key, value in job:
            if key == 'env':
                return value
        raise ValueError("job is missing --env")

    def _run(self, cmd, logdir):
        with self.lock:
            if not os.path.exists(logdir):
                os.mkdir(logdir)

            overwrite = not self._is_complete(logdir)
            logfile = os.path.join(logdir, 'log.txt')

            print(' '.join(cmd))
            print('> ' + logfile)
            if not overwrite:
                print("(run already complete)")
            print(flush=True)

        if overwrite:
            self._exec_and_wait(cmd, logfile)

    def _is_complete(self, logdir):
        return os.path.exists(os.path.join(logdir, 'OK'))

    def _exec_and_wait(self, cmd, logfile):
        with open(logfile, 'w') as f:
            p = Popen(cmd, stdout=f, stderr=f)
            p.wait()


def cartesian_product(config):
    jobs = [[]]
    for name, values in config.items():
        # If there is only one value, wrap it in a list
        if not isinstance(values, list):
            values = [values]
        # Generate the product of the experiments we have so far with the new values
        for _ in range(len(jobs)):
            j = jobs.pop(0)
            for v in values:
                jobs.append(j + [(name, str(v))])
    return jobs


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='logs/')
    args = parser.parse_args()


    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Change cap size and layer size and goal_selection_strategy  in yaml file. Keep log dir here.
    ENV_DEFAULTS = {
        'cheese': f'--logdir {output_dir}/cheese',
        'cheeseonehot': f'--logdir {output_dir}/cheeseonehot',
        'hallway': f'--logdir {output_dir}/hallway',
        'hallwayonehot': f'--logdir {output_dir}/hallwayonehot',
        'mit': f'--logdir {output_dir}/mit',
        'mitonehot': f'--logdir {output_dir}/mitonehot',
        'cit': f'--logdir {output_dir}/citonehot',
    }



    with open('automate_config.yaml', 'r') as fh:
        config = yaml.safe_load(fh)

    dispatcher = Dispatcher(config, output_dir)
    dispatcher.execute()
