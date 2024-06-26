import pyspiel
import torch
import sys
sys.path.append("/workspace/src/")
import env_headsup
from Curriculum import Curriculum
import enum
import numpy as np
import inspect
import copy

_NUM_PLAYERS = 2
_GAME_TYPE = pyspiel.GameType(
    short_name="HeadsUpGame",
    long_name="Python HeadsUpGame",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=12,
    max_chance_outcomes=52,
    num_players=_NUM_PLAYERS,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=20)  # e.g. Pass, Bet, Bet

convert_num_to_action = {
    0: "fold",
    1: "call",
    2: "check",
    3: "raise"
}


class HaeadsUpGame(pyspiel.Game):
    """A Python version of HeadsUpGame."""

    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        config_path = "/workspace/src/config/config.json"
        opp_mode = "random"
        is_train = False
        self_model_path = ""
        seed = 0
        curriculum = Curriculum(config_path)

        gym_env = env_headsup.Headsup(
            config_path=config_path,
            opp_mode=opp_mode,
            is_train=is_train,
            self_model_path=self_model_path,
            seed=seed,
            curriculum=curriculum,
            one_hand=True
            )
        self.gym_env = gym_env

    def new_initial_state(self):
        """Returns a state corresponding to the start of a game."""
        return HeadsUpGameState(self)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return HeadsUpGameObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
            params)

class HeadsUpGameState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.game = game.gym_env
        self.game.reset()

        self._game_over = False
        self.reward = 0
        self.action_space_num = game.gym_env.action_space_num
        self.info = None
        self.get_pot = 0
        self.history = []

    def clone(self):
        new_state = copy.deepcopy(self)
        return new_state

    def child(self, action):
        new_state = self.clone()
        new_state._apply_action(action)
        return new_state

    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        else:
            if self.game.game_status.current_player == "A":
                return 0
            else:
                return 1
    def history_str(self):
        return " ".join([str(i) for i in self.history])

    def _legal_actions(self, player):
        assert player >=0
        self.game.get_observation()
        mask = self.game.observation["mask"]
        # 全てFalseの場合はprintして終了
        assert any(mask), "mask is all False"
        legal_list = [i for i in range(self.action_space_num) if mask[i]]
        if len(legal_list) < 2:
            print("mask", mask)
            print("observation", self.game.observation)
            print("game_status", self.game.game_status)
            print("player", player)
            print("current_player", self.current_player())
            print("info", self.info)
            print("reward", self.reward)
            print("game_over", self._game_over)
            print("legal_list", legal_list)

        return legal_list

    def legal_actions(self, player=0):
        self.game.get_observation()
        mask = self.game.observation["mask"]
        # 全てFalseの場合はprintして終了
        assert any(mask), "mask is all False"
        legal_list = [i for i in range(self.action_space_num) if mask[i]]
        if len(legal_list) < 2:
            print("mask", mask)
            print("observation", self.game.observation)
            print("game_status", self.game.game_status)
            print("player", player)
            print("current_player", self.current_player())
            print("info", self.info)
            print("reward", self.reward)
            print("game_over", self._game_over)
            print("legal_list", legal_list)
        """
        print("----")
        print(self)
        print("legal_list", legal_list)
        print("----")
        """
        return legal_list
    
    def chance_outcomes(self):
        assert self.is_chance_node()

        return [(i, 1/self.action_space_num) for i in range(self.action_space_num)]
    
    def _apply_action(self, action):
        if self.is_chance_node():
            pass
        else:
            if self._game_over:
                assert False, "game is over"
            self.history.append(action)
            observation, all_reward, done, _, info = self.game.one_step(action)
            self.info = info
            self.reward = all_reward
            self._game_over = done
            if self._game_over:
                self.get_pot = info["get_pot"]

    def _action_to_string(self, player, action):
        if action < 3:
            return convert_num_to_action[action]
        else:
            return f"raise_{action-2}"
    
    def is_terminal(self):
        # print("is_terminal", self._game_over)
        return self._game_over

    def returns(self):
        # reward = self.get_pot / 4000.0
        self.game.get_observation()
        reward_a = (self.game.game_status.Bankroll_A - self.game.game_status.Start_Bankroll_A) / 2000.0
        reward_b = (self.game.game_status.Bankroll_B - self.game.game_status.Start_Bankroll_B) / 2000.0 

        if not self._game_over:
            return [0, 0]
        else:
            return [reward_a, reward_b]
        """
        elif self.game.game_status.winner == "A":
            return [reward, -reward]
        else:
            return [-reward, reward]
        """
    
    def information_state_tensor(self, player=None):
        # _tmp = self.game.observation["vector"]
        self.game.get_observation()
        _tmp = self.game.observation_vector

        return torch.tensor(_tmp, dtype=torch.float32)
        
    def __str__(self):
        _str = f"game_state: {self.game.game_status.game_state} current_player: {self.game.game_status.current_player}\n"
        _str += f"position_A: {self.game.game_status.position_A} position_B: {self.game.game_status.position_B}\n"
        _str += f"pot: {self.game.game_status.pot} Bankroll_A: {self.game.game_status.Bankroll_A} Bankroll_B: {self.game.game_status.Bankroll_B}\n"
        _str += f"bet_A: {self.game.game_status.bet_A} bet_B: {self.game.game_status.bet_B}\n"
        _str += f"hands_A: {self.game.game_status.hands_A} hands_B: {self.game.game_status.hands_B}\n"
        _str += f"board_cards: {self.game.game_status.board_cards}\n"
        _str += f"action A: {self.game.action_A} B: {self.game.action_B}\n"
        _str += f"action_history A: {self.game.game_status.action_history_A} B: {self.game.game_status.action_history_B}\n"
        _str += f"done: {self._game_over}\n"
        _str += f"winnner: {self.game.game_status.winner}\n"
        if self.game.game_status.winner is not None:
            _str += f"get_pot: {self.get_pot}\n"
        _str += f"env_id: {self.game.env_id}"

        return _str
   
class HeadsUpGameObserver:
    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        """
        pieces = [("my_hand_vec", 17, (17,))]
        pieces.append(("public_cards", 17 * 5, (17 *5, )))
        """
        pieces = [("my_hand_vec", 2, (2,))]
        pieces.append(("public_cards", 5, (5, )))
        pieces.append(("pot", 1, (1,)))
        pieces.append(("my_pos", 1, (1,)))
        pieces.append(("my_bankroll", 1, (1,)))
        pieces.append(("opp_bankroll", 1, (1,)))
        pieces.append(("my_bet", 1, (1,)))
        pieces.append(("opp_bet", 1, (1,)))
        pieces.append(("my_action_history", 10, (10,)))
        pieces.append(("opp_action_history", 10, (10,)))

        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)
        self.dict = {}
        index = 0
        for name, size, shape in pieces:
            self.dict[name] = self.tensor[index:index+size].reshape(shape)
            index += size

    def set_from(self, state, player):
        self.tensor.fill(0)
        # state.game.get_observation()
        _tmp = state.game.observation_vector
        self.dict["my_hand_vec"][:] = _tmp[0:2]
        self.dict["public_cards"][:] = _tmp[2:7]
        self.dict["pot"][:] = _tmp[7]
        self.dict["my_pos"][:] = _tmp[8]
        self.dict["my_bankroll"][:] = _tmp[9]
        self.dict["opp_bankroll"][:] = _tmp[10]
        self.dict["my_bet"][:] = _tmp[11]
        self.dict["opp_bet"][:] = _tmp[12]
        self.dict["my_action_history"][:] = _tmp[13:23]
        self.dict["opp_action_history"][:] = _tmp[23:33]

        """
        self.dict["my_hand_vec"][:] = _tmp[0:17]
        self.dict["public_cards"][:] = _tmp[17:119]
        self.dict["pot"][:] = _tmp[119]
        self.dict["my_pos"][:] = _tmp[120]
        self.dict["my_bankroll"][:] = _tmp[121]
        self.dict["opp_bankroll"][:] = _tmp[122]
        self.dict["my_bet"][:] = _tmp[123]
        self.dict["opp_bet"][:] = _tmp[124]
        self.dict["my_action_history"][:] = _tmp[125:245]
        self.dict["opp_action_history"][:] = _tmp[245:365]

        print("set_from")
        _tmp = state.game.observation["vector"]
        self.dict["opp_action_vec"][:] = _tmp[0:16]
        self.dict["pot"][:] = _tmp[16]
        self.dict["my_pos"][:] = _tmp[17]
        self.dict["my_bankroll"][:] = _tmp[18]
        self.dict["opp_bankroll"][:] = _tmp[19]
        self.dict["my_bet"][:] = _tmp[20]
        self.dict["opp_bet"][:] = _tmp[21]
        self.dict["opp_raise_size"][:] = _tmp[22]
        self.dict["state"][:] = _tmp[23:27]
        self.dict["my_hand"][:] = _tmp[27:32]
        self.dict["public_cards"][:] = _tmp[32:49]
        self.dict["relation_vec"][:] = _tmp[49:57]
        """


    def string_from(self, state, player):
        return "string from is not implemented."



pyspiel.register_game(_GAME_TYPE, HaeadsUpGame)


if __name__=="__main__":
    pass