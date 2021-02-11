import re

from pkg_resources import resource_exists, resource_filename, resource_listdir
import numpy as np

from .envs import *
from .envs.registration import env_list, register, register_mdp
from .wrappers import *

__version__ = '1.0.0'

extension = 'pomdp'


def list_pomdps():
    return list(env_list)


def is_pomdp(filename):  # pylint: disable=redefined-outer-name
    return filename.casefold().endswith(
        f'.{extension.casefold()}'
    ) and resource_exists('gym_pomdps.pomdps', filename)


for filename in (
    filename
    for filename in resource_listdir('gym_pomdps', 'pomdps')
    if filename.casefold().endswith(f'.{extension.casefold()}')
):
    path = resource_filename('gym_pomdps.pomdps', filename)
    name, _ = filename.rsplit('.', 1)  # remove .pomdp extension
    version = 0

    # extract version if any
    m = re.fullmatch(r'(?P<name>.*)\.v(?P<version>\d+)', name)
    if m is not None:
        name, version = m['name'], int(m['version'])

    with open(path) as f:
        text = f.read()

    #register(
    #    id=f'POMDP-{name}-continuing-v{version}',
    #    entry_point='gym_pomdps.envs:POMDP',
    #    kwargs=dict(text=text, episodic=False),
    #)

    register(
        id=f'POMDP-{name}-episodic-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(text=text, episodic=True),
    )
    for env_name in ['cheese', 'hallway', 'bigcheese']:
        if filename == f'{env_name}.pomdp':
            register(
                id=f'POMDP-{env_name}onehot-v{version}',
                entry_point=f'gym_pomdps.envs:{env_name.capitalize()}OneHotPOMDP',
                kwargs=dict(text=text, step_cap=np.inf)
            )
        elif filename == f'{env_name}_goal.pomdp':
            register(
                id=f'POMDP-Goal-{env_name}onehot-v{version}',
                entry_point=f'gym_pomdps.envs:{env_name.capitalize()}OneHotGoalPOMDP',
                kwargs=dict(text=text, step_cap=np.inf)
            )
            register(
                id=f'POMDP-Goalconcat-{env_name}onehot-v{version}',
                entry_point=f'gym_pomdps.envs:{env_name.capitalize()}OneHotConcatPOMDP',
                kwargs=dict(text=text, step_cap=np.inf)
            )
    for env_name in ['hallway', 'cheese', 'bigcheese', 'mit', 'cit']:
        if filename == env_name + 'mdp.pomdp':
            register_mdp(
                id=f'MDP-{env_name}-v{version}',
                entry_point=f'gym_pomdps.envs:{env_name.capitalize()}MDP',
                kwargs=dict(text=text, step_cap=np.inf)
            )
            register_mdp(
                id=f'MDP-{env_name}onehot-v{version}',
                entry_point=f'gym_pomdps.envs:{env_name.capitalize()}OneHotMDP',
                kwargs=dict(text=text, step_cap=np.inf)
            )
