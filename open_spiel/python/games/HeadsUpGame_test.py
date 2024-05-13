import pyspiel
import torch
import sys
sys.path.append("../")
import env_headsup
from Curriculum import Curriculum

class HeadupGameAdapter(pyspiel.Game):
    def __init__(self, config_path, opp_mode, is_train, self_model_path, seed, curriculum):
        gym_env = env_headsup.Headsup(
            config_path=config_path,
            opp_mode=opp_mode,
            is_train=is_train,
            self_model_path=self_model_path,
            seed= seed,
            curriculum=curriculum
            )
        self.gym_env = gym_env
        self._state = gym_env.reset()
        self._state_string = str(self._state)
        self._num_players = 2
        self._num_distinct_actions = 2
        


if __name__=="__main__":
    config_path = "../config/config.json"
    opp_mode = "random"
    is_train = True
    self_model_path = ""
    seed = 0
    curriculum = Curriculum(config_path)

