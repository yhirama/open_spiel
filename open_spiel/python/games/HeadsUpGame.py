import pyspiel
import torch
import sys
sys.path.append("/workspace/src/")
import env_headsup
from Curriculum import Curriculum
import enum
import numpy as np
import inspect

_NUM_PLAYERS = 2
_DECK = frozenset([0, 1, 2])
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
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-2.0,
    max_utility=2.0,
    utility_sum=0.0,
    max_game_length=3)  # e.g. Pass, Bet, Bet

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
    
    def current_player(self):
        if self._game_over:
            return pyspiel.PlayerId.TERMINAL
        else:
            if self.game.game_status.current_player == "A":
                return 0
            else:
                return 1

    def _legal_actions(self, player):
        assert player >=0
        # self.game.get_observation()
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
        # self.game.get_observation()
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
    
    def chance_outcomes(self):
        assert self.is_chance_node()

        return [(i, 1/self.action_space_num) for i in range(self.action_space_num)]
    
    def _apply_action(self, action):
        if self.is_chance_node():
            pass
        else:
            if self._game_over:
                assert False, "game is over"
            print("apply_action")
            print("current_player", self.game.game_status.current_player)
            observation, all_reward, done, _, info = self.game.one_step(action)
            self.info = info
            self.reward = all_reward
            self._game_over = done

    def _action_to_string(self, player, action):
        if action < 3:
            return convert_num_to_action[action]
        else:
            return f"raise_{action-3}"
    
    def is_terminal(self):
        # print("is_terminal", self._game_over)
        return self._game_over

    def returns(self):
        _pot = self.game.game_status.pot + self.game.game_status.bet_A + self.game.game_status.bet_B
        if not self._game_over:
            return [0, 0]
        elif self.game.game_status.winner == "A":
            return [1, -1]
        else:
            return [-1, 1]
        
    def __str__(self):
        _str = f"game_state: {self.game.game_status.game_state} current_player: {self.game.game_status.current_player}\n"
        _str += f"position_A: {self.game.game_status.position_A} position_B: {self.game.game_status.position_B}\n"
        _str += f"pot: {self.game.game_status.pot} Bankroll_A: {self.game.game_status.Bankroll_A} Bankroll_B: {self.game.game_status.Bankroll_B}\n"
        _str += f"bet_A: {self.game.game_status.bet_A} bet_B: {self.game.game_status.bet_B}\n"
        _str += f"hands_A: {self.game.game_status.hands_A} hands_B: {self.game.game_status.hands_B}\n"
        _str += f"board_cards: {self.game.game_status.board_cards}\n"
        _str += f"done: {self._game_over}\n"
        _str += f"winnner: {self.game.game_status.winner}\n"
        _str += f"env_id: {self.game.env_id}\n"

        return _str
   
class HeadsUpGameObserver:
    def __init__(self, iig_obs_type, params):
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")
        """
        pieces = [("player", 2, (2,))]
        if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
            pieces.append(("private_cards",34 , (2,17)))
        pieces.append(("public_cards", 85, (5, 17))) 
        total_size = sum(size for name, size, shape in pieces)
        self.tensor = np.zeros(total_size, np.float32)

        """
        pieces = [("opp_action_vec", 16, (16,))]
        pieces.append(("pot", 1, (1,)))
        pieces.append(("my_pos", 1, (1,)))
        pieces.append(("my_bankroll", 1, (1,)))
        pieces.append(("opp_bankroll", 1, (1,)))
        pieces.append(("my_bet", 1, (1,)))
        pieces.append(("opp_bet", 1, (1,)))
        pieces.append(("opp_raise_size", 1, (1,)))
        pieces.append(("state", 4, (4,)))
        pieces.append(("my_hand", 5, (5, )))
        pieces.append(("public_cards", 17, (17, )))
        pieces.append(("relation_vec", 8, (8,)))

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
        if "player" in self.dict:
            self.dict["player"][player] = 1
        if "private_cards" in self.dict:
            self.dict["private_cards"][:] = state.game.get_observation_dict("private_cards")
        if "public_cards" in self.dict:
            self.dict["public_cards"][:] = state.game.get_observation_dict("public_cards")
        """

    def string_from(self, state, player):
        return "string from is not implemented."



pyspiel.register_game(_GAME_TYPE, HaeadsUpGame)


if __name__=="__main__":
    pass