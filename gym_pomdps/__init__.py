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

    register(
        id=f'POMDP-{name}-continuing-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(text=text, episodic=False),
    )

    register(
        id=f'POMDP-{name}-episodic-v{version}',
        entry_point='gym_pomdps.envs:POMDP',
        kwargs=dict(text=text, episodic=True),
    )
    if filename == 'cheese_goal.pomdp':
        register(
        id=f'POMDP-Goal-cheeseonehot-v{version}',
        entry_point='gym_pomdps.envs:CheeseOneHotPOMDP',
        kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
    if filename == 'hallway_goal.pomdp':
        register(
            id=f'POMDP-Goal-hallwayonehot-v{version}',
            entry_point='gym_pomdps.envs:HallwayOneHotPOMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
    # Okay this is gross but it's what I'm working with right now.
    if filename == 'hallwaymdp.pomdp':
        # Simpler name to help on the old fingers, ya feel? This is the same as
        # MDP-hallwaymdp-dense-episodic-v0
        register_mdp(
            id=f'MDP-hallway-v{version}',
            entry_point='gym_pomdps.envs:HallwayMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        # integer version of Hallway, dense and sparse.
        register_mdp(
            id=f'MDP-hallwaymdp-dense-episodic-v{version}',
            entry_point='gym_pomdps.envs:HallwayMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-hallwaymdp-sparse-episodic-v{version}',
            entry_point='gym_pomdps.envs:HallwayMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf,  dense_reward=False),
        )
        # One hot version of Hallway
        register_mdp(
            id=f'MDP-hallwayonehot-v0',
            entry_point='gym_pomdps.envs:HallwayOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-hallwayonehotmdp-dense-episodic-v0',
            entry_point='gym_pomdps.envs:HallwayOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-hallwayonehotmdp-sparse-episodic-v0',
            entry_point='gym_pomdps.envs:HallwayOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf,  dense_reward=False)
        )
    elif filename == 'mitmdp.pomdp':
        register_mdp(
            id=f'MDP-mit-v{version}',
            entry_point='gym_pomdps.envs:MITMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-mitmdp-dense-episodic-v{version}',
            entry_point='gym_pomdps.envs:MITMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-mitmdp-sparse-episodic-v{version}',
            entry_point='gym_pomdps.envs:MITMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf, dense_reward=False),
        )
        register_mdp(
            id=f'MDP-mitonehot-v{version}',
            entry_point='gym_pomdps.envs:MITOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-mitonehotmdp-dense-episodic-v{version}',
            entry_point='gym_pomdps.envs:MITOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-mitonehotmdp-sparse-episodic-v{version}',
            entry_point='gym_pomdps.envs:MITOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf, dense_reward=False)
        )
    elif filename == 'cheesemdp.pomdp':
        register_mdp(
            id=f'MDP-cheese-v{version}',
            entry_point='gym_pomdps.envs:CheeseMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-cheesemdp-dense-episodic-v{version}',
            entry_point='gym_pomdps.envs:CheeseMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-cheesemdp-sparse-episodic-v{version}',
            entry_point='gym_pomdps.envs:CheeseMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf, dense_reward=False),
        )
        register_mdp(
            id=f'MDP-cheeseonehot-v0',
            entry_point='gym_pomdps.envs:CheeseOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-cheeseonehotmdp-dense-episodic-v0',
            entry_point='gym_pomdps.envs:CheeseOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf),
        )
        register_mdp(
            id=f'MDP-cheeseonehotmdp-sparse-episodic-v0',
            entry_point='gym_pomdps.envs:CheeseOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf, dense_reward=False),
        )
    elif filename == 'citmdp.pomdp':
        register_mdp(
            id=f'MDP-cit-v0',
            entry_point='gym_pomdps.envs:CITMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-citmdp-dense-episodic-v0',
            entry_point='gym_pomdps.envs:CITMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-citmdp-sparse-episodic-v0',
            entry_point='gym_pomdps.envs:CITMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf, dense_reward=False)
        )
        register_mdp(
            id=f'MDP-citonehot-v0',
            entry_point='gym_pomdps.envs:CITOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-citonehotmdp-dense-episodic-v0',
            entry_point='gym_pomdps.envs:CITOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf)
        )
        register_mdp(
            id=f'MDP-citonehotmdp-sparse-episodic-v0',
            entry_point='gym_pomdps.envs:CITOneHotMDP',
            kwargs=dict(text=text, episodic=True, step_cap=np.inf, dense_reward=False)
        )
    else:
        register_mdp(
            id=f'MDP-{name}-continuing-v{version}',
            entry_point='gym_pomdps.envs:MDP',
            kwargs=dict(text=text, episodic=False),
        )
        register_mdp(
            id=f'MDP-{name}-episodic-v{version}',
            entry_point='gym_pomdps.envs:MDP',
            kwargs=dict(text=text, episodic=True),
        )


