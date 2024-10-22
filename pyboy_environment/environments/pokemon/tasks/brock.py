from functools import cached_property

import numpy as np
from pyboy.utils import WindowEvent

from pyboy_environment.environments.pokemon.pokemon_environment import (
    PokemonEnvironment,
)
from pyboy_environment.environments.pokemon import pokemon_constants as pkc

# Reward Constants
BASE_REWARD = -1
START_BATTLE_REWARD = 10
DEAL_DAMAGE_MULTIPLIER = 10
GAIN_XP_MULTIPLER = 1
LEVEL_UP_MULTIPLIER = 10000

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
        return {
            **self._get_location(),
            "in_grass": self._in_grass_tile(),
            "party_size": self._get_party_size(),
            "ids": self._read_party_id(),
            # "pokemon": [pkc.get_pokemon(id) for id in self._read_party_id()],
            "levels": self._read_party_level(),
            "type_id": self._read_party_type(),
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
        }
    
    # OVERRIDE
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
    
    def _get_current_pokemon_id(self) -> int:
        return self._read_m(0xD014)
    
    def _in_grass_tile(self):
        if self._is_grass_tile():
            return 1
        else:
            return 0

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
        return damage_dealt * DEAL_DAMAGE_MULTIPLIER
    
    def _xp_reward(self, new_state: dict[str, any]) -> int:
        delta_xp = sum(new_state["xp"]) - sum(self.prior_game_stats["xp"])
        return delta_xp * GAIN_XP_MULTIPLER

    def _levels_reward(self, new_state: dict[str, any]) -> float:
        reward = 0
        new_levels = new_state["levels"]
        old_levels = self.prior_game_stats["levels"]
        for i in range(len(new_levels)):
            if (old_levels[i] != 0):
                reward += (new_levels[i] / old_levels[i] - 1) * LEVEL_UP_MULTIPLIER
        return reward

    def _calculate_reward(self, new_state: dict) -> float:
        reward = BASE_REWARD
        reward += self._start_battle_reward(new_state)
        reward += self._deal_damage_reward(new_state)
        reward += self._xp_reward(new_state)
        reward += self._levels_reward(new_state)
        return reward

    def _check_if_done(self, game_stats: dict[str, any]) -> bool:
        # Setting done to true if agent beats first gym (temporary)
        return game_stats["badges"] > self.prior_game_stats["badges"]

    def _check_if_truncated(self, game_stats: dict) -> bool:
        # Implement your truncation check logic here

        # Maybe if we run out of pokeballs...? or a max step count
        return self.steps >= 1000
