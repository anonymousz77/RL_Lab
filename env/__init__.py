import gymnasium as gym

from .historicalobs import *
from .doorkey import *
from .lavadoorkey import *
from .twodoor import *
from .coloreddoorkey import *


gym.register(
    id='MiniGrid-SimpleDoorKey-Min5-Max10-View3',
    entry_point='env.doorkey:DoorKeyEnv',
    kwargs={'minRoomSize' : 5,
            'maxRoomSize' : 10,
            'agent_view_size' : 3,
            'max_steps': 150},
)

gym.register(
    id='MiniGrid-LavaDoorKey-Min5-Max10-View3',
    entry_point='env.lavadoorkey:LavaDoorKeyEnv',
    kwargs={'minRoomSize' : 5,
            'maxRoomSize' : 10,
            'agent_view_size' : 3,
            'max_steps': 150},
)

gym.register(
    id='MiniGrid-ColoredDoorKey-Min5-Max10-View3',
    entry_point='env.coloreddoorkey:ColoredDoorKeyEnv',
    kwargs={'minRoomSize' : 5,
            'maxRoomSize' : 10,
            'minNumKeys' : 2,
            'maxNumKeys' : 2,
            'agent_view_size' : 3,
            'max_steps' : 150},
)

gym.register(
    id='MiniGrid-TwoDoor-Min17-Max20',
    entry_point='env.twodoor:TwoDoorEnv',
    kwargs={'minRoomSize' : 17,
            'maxRoomSize' : 20,
            'agent_view_size' : 3,
            'max_steps' : 150},
)