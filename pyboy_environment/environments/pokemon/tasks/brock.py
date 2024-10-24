from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

# Reward Constants
# Larger value giving to sparser rewards
# Smaller value giving to more frequently experiened rewards
BASE_REWARD = -2
IN_GRASS_REWARD = 1
START_BATTLE_REWARD = 100
DEAL_DAMAGE_MULTIPLIER = 100
GAIN_XP_MULTIPLER = 10
LEVEL_UP_MULTIPLIER = 10000
MOVE_UP_REWARD = 1
ENTER_POKEMART_REWARD = 1000
PURCHASE_POKEBALL_MULTIPLIER = 500
CATCH_POKEMON_REWARD = 1000
TASK_COMPLETION_MULTIPLIER = 10000
MOVE_CLOSER_TO_GYM_REWARD = 10000

STEPS_TRUNCATION = 500
TASK_COMPLETION_EXTRA_STEPS = 600
LEVEL_UP_EXTRA_STEPS = 200
FIND_BROCK_EXTRA_STEPS = 200

class PokemonBrock(PokemonEnvironment):
    def __init__(
        self,
        act_freq: int,
        emulation_speed: int = 0,
        headless: bool = False,
    ) -> None:

        valid_actions: list[WindowEvent] = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]

        release_button: list[WindowEvent] = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        super().__init__(
            act_freq=act_freq,
            task="brock",
            init_name="has_pokedex.state",
            emulation_speed=emulation_speed,
            valid_actions=valid_actions,
            release_button=release_button,
            headless=headless,
        )

    def _generate_game_stats(self) -> dict[str, any]:
        game_stats = {
            **self._get_location(),
            "in_grass": self._is_in_grass(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            # "pokemon": [pkc.get_pokemon(id) for id in self._read_party_id()],
            "levels": self._read_party_level(),
            # "type_id": self._read_party_type(),
            # "type": [pkc.get_type(id) for id in self._read_party_type()],
            **self._read_party_hp(),
            "xp": self._read_party_xp(),
            "status": self._read_party_status(),
            "badges": self._get_badge_count(),
            # "caught_pokemon": self._read_caught_pokemon_count(),
            # "seen_pokemon": self._read_seen_pokemon_count(),
            "money": self._read_money(),
            # "events": self._read_events(),
            "battle_type": self._read_battle_type(),
            "enemy_pokemon_health": self._get_enemy_pokemon_health(),
            "current_pokemon_id": self._get_current_pokemon_id(),
            "num_pokeballs": self._get_num_pokeballs(),
            "current_selected_menu_item": self._get_current_selected_menu_item()
        }
        
        game_stats["tasks"] = self._get_task_list(game_stats)

        return game_stats
    
    # Represent tasks as individual 0/1 inputs
    def _get_task_list(self, game_stats: dict) -> list[int]:
        active_task = self._select_task(game_stats)
        num_tasks = 8
        tasks = [0] * num_tasks
        tasks[active_task] = 1
        return tasks
    
    def _select_task(self, game_stats: dict) -> int:
        if (game_stats["levels"][0] < 12):
            return 1 # fight
        elif (game_stats["party_size"] < 3 and
              self._get_num_pokeballs(game_stats) < 10
            ):
            if (game_stats["map_id"] != 1 and game_stats["map_id"] != 0x2a):
                return 2 # enter pokemart village
            elif(game_stats["map_id"] == 1):
                return 3 # enter pokemart
            elif(game_stats["map_id"] == 0x2a):
                return 4 # purchase pokeballs
        elif (game_stats["party_size"] < 3):
            return 5 # catch pokemon
        elif (self._assert_min_pokemon_level(game_stats, 4)):
            return 6 # train party to specified level
        elif (game_stats["map_id"] != 0x36):
            return 7 # find gym
        else:
            return 0 # defeat brock
        
    def _assert_min_pokemon_level(self, game_stats: dict, min: int) -> bool:
        num_pokemon = game_stats["party_size"]
        levels = game_stats["levels"]

        for i in range(num_pokemon):
            if (levels[i] < min):
                return False
        
        return True
         
    # OVERRIDE to remove map name (string)
    def _get_location(self) -> dict[str, any]:
        x_pos = self._read_m(0xD362)
        y_pos = self._read_m(0xD361)
        map_n = self._read_m(0xD35E)

        return {
            "x": x_pos,
            "y": y_pos,
            "map_id": map_n,
        }

    def _read_battle_type(self) -> int:
        return self._read_m(0xD057)

    def _get_enemy_pokemon_health(self) -> int:
        return self._read_hp(0xCFE6)
    
    def _get_current_selected_menu_item(self) -> int:
        return self._read_m(0xCC26)
    
    def _get_index_current_pokemon(self) -> int:
        return self._read_m(0xCC2F)
    
    def _get_current_pokemon_id(self) -> int:
        return self._read_m(0xD014)
    
    def _is_in_grass(self) -> int:
        if self._is_grass_tile():
            return 1
        else:
            return 0
        
    # returns a dictionary of owned items (can't be used in state space as the size of the dictionary may vary)
    def _get_items(self) -> dict:
        total_items = self._read_m(0xD31D)
        if (total_items == 0):
            return {}

        addr = 0xD31E
        items = {}

        for i in range(total_items):
            item_id = self._read_m(addr + 2 * i)
            item_count = self._read_m(addr + 2 * i + 1)
            items[f"item_{item_id}"] = item_count

        return items
    
    # computes the total number of poke balls.
    def _get_num_pokeballs(self) -> int:
        items = self._get_items()
        keys = items.keys()
        num_pokeballs = 0

        for i in range(0x5):
            key = f"item_{i}"
            if (key in keys):
                num_pokeballs += items[key]

        return num_pokeballs

    def _get_state(self) -> np.ndarray:
        game_stats = self._generate_game_stats()
        state = self._get_state_from_stats(game_stats)
        return state
    
    def _get_state_from_stats(self, game_stats: dict) -> np.ndarray:
        state = []
        for value in game_stats.values():
            if isinstance(value, list):
                for i in value:
                    state.append(i)                      
            else:
                state.append(value)
        return np.array(np.array(state))
    
    def _start_battle_reward(self, new_state) -> int:
        if (new_state["battle_type"] != 0 and self.prior_game_stats["battle_type"] == 0):
            return START_BATTLE_REWARD
        return 0
    
    def _deal_damage_reward(self, new_state: dict[str, any]) -> int:
        damage_dealt = self.prior_game_stats["enemy_pokemon_health"] - new_state["enemy_pokemon_health"]
        if (new_state["battle_type"] != self.prior_game_stats["battle_type"]):
            return 0
        else:
            return damage_dealt * DEAL_DAMAGE_MULTIPLIER
    
    def _xp_reward(self, new_state: dict[str, any]) -> int:
        delta_xp = sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])
        return delta_xp * GAIN_XP_MULTIPLER

    def _levels_reward(self, new_state: dict[str, any]) -> float:
        reward = 0
        new_levels = new_state["levels"]
        old_levels = self.prior_game_stats["levels"]
        for i in range(len(new_levels)):
            if (new_levels[i] > old_levels[i]):
                reward += (new_levels[i] / old_levels[i] - 1) * LEVEL_UP_MULTIPLIER
                self.steps -= LEVEL_UP_EXTRA_STEPS # add extra hundred steps after a level up is performed
        return reward
    
    def _get_fight_reward(self, new_state: dict) -> float:
        reward = self._is_in_grass() * IN_GRASS_REWARD
        reward += self._start_battle_reward(new_state)
        reward += self._deal_damage_reward(new_state)
        reward += self._xp_reward(new_state)
        reward += self._levels_reward(new_state)
        return reward
    
    def _get_enter_pokemart_village_reward(self, new_state: dict) -> float:

        if (new_state["y"] < self.prior_game_stats["x"] or
            (new_state["map_id"] == 0x0c and self.prior_game_stats["map_id"] == 00)):
            return MOVE_UP_REWARD
        
        return 0
    
    def _get_enter_pokemart_reward(self, new_state: dict) -> float:
        if (self.prior_game_stats["map_id"] != 0x2a and new_state["map_id"] == 0x2a):
            return ENTER_POKEMART_REWARD
        else:
            return 0

    def _get_purchase_pokeballs_reward(self, new_state: dict) -> float:
        delta_pokeball = self._get_num_pokeballs(new_state) - self._get_num_pokeballs(self.prior_game_stats)
        if (delta_pokeball > 0):
            return delta_pokeball * PURCHASE_POKEBALL_MULTIPLIER
        return 0

    def _get_catch_pokemon_reward(self, new_state: dict) -> float:
        delta_pokeball = self._get_num_pokeballs(new_state) - self._get_num_pokeballs(self.prior_game_stats)
        reward = 0        
        if (delta_pokeball < 0):
            reward += delta_pokeball * -PURCHASE_POKEBALL_MULTIPLIER
        
        if (new_state["party_size"] > self.prior_game_stats["party_size"]):
            reward += CATCH_POKEMON_REWARD

        return reward

    def _find_gym_reward(self, new_state: dict) -> float:
        room_ids = [0x00, 0x0c, 0x01, 0x0d, 0x32, 0x33, 0x2f, 0x0d, 0x02, 0x36]
        old_index = room_ids.index(self.prior_game_stats["map_id"])
        new_index = room_ids.index(new_state["map_id"])

        if (new_index > old_index):
            self.steps -= FIND_BROCK_EXTRA_STEPS
            return MOVE_CLOSER_TO_GYM_REWARD
        elif (new_index < old_index):
            if (self.prior_game_stats["map_id"] == 0x2f and new_state["map_id"] == 0x0d):
                self.steps -= FIND_BROCK_EXTRA_STEPS
                return MOVE_CLOSER_TO_GYM_REWARD # Edge case as 0x0d is found twice
            else:
                return -MOVE_CLOSER_TO_GYM_REWARD
        else:
            return 0

    def _get_fight_brock_reward(self, new_state: dict) -> float:
        reward = 0
        if (new_state["battle_type" ==  2]):
            if (self.prior_game_stats["battle_type"] == 2):
                reward += 5
            else:
                reward += 100
        
        reward += self._deal_damage_reward(new_state)

    def _get_change_in_task(self, new_state: dict) -> float:
        old_task = self.prior_game_stats["tasks"].index(1)
        new_task = new_state["tasks"].index(1)

        if (new_task > old_task):
            self.steps -= TASK_COMPLETION_EXTRA_STEPS

        return (new_task - old_task) * TASK_COMPLETION_MULTIPLIER

    def _calculate_reward(self, new_state: dict) -> float:
        reward = BASE_REWARD

        # compute the reward for the new state given the previous task and the corresponding action
        task = self.prior_game_stats["tasks"].index(1)
        if (task == 1 or task == 6):
            reward += self._get_fight_reward(new_state)
        elif (task == 2):
            reward += self._get_enter_pokemart_village_reward(new_state)
        elif (task == 3):
            reward += self._get_enter_pokemart_reward(new_state)
        elif (task == 4):
            reward += self._get_purchase_pokeballs_reward(new_state)
        elif (task == 5):
            reward += self._get_catch_pokemon_reward(new_state)
        elif (task == 7):
            reward += self._find_gym_reward(new_state)
        else:
            reward += self._get_fight_brock_reward(new_state)

        # reward for completing a task (is negative for reverting to previous task)
        reward += self._get_change_in_task(new_state)

        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        return self.steps >= STEPS_TRUNCATION