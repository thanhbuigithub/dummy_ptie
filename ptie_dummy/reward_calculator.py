"""
Oracle Reward Calculator for PTIE Dummy Implementation
Implements perfect information reward calculation following PerfectDou's design.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from game.game import DummyGame, GameState
from game.player import Player, PlayerIndex
from game.card import Card, CardList
from game.form import Form, CARDS_MAP


class OracleRewardCalculator:
    """
    Calculates oracle-based rewards using perfect information.
    
    Following PerfectDou's approach:
    - Uses dynamic programming to calculate minimum steps to win
    - Provides dense reward signal based on advantage changes
    - Encourages cooperation between team players
    """
    
    def __init__(self, reward_scaling: float = 1.0):
        self.reward_scaling = reward_scaling
        self.memo = {}  # Memoization for DP calculations
        
    def calculate_oracle_reward(self, game: DummyGame, current_player: Player) -> float:
        """
        Calculate oracle reward for the current game state.
        
        Args:
            game: Current game instance
            current_player: Player for whom to calculate reward
            
        Returns:
            Oracle reward value
        """
        if game.game_state != GameState.STARTED:
            return 0.0
        
        # Calculate minimum steps to win for all players
        min_steps = {}
        for player in game.players:
            min_steps[player.id] = self._calculate_min_steps_to_win(player, game)
        
        # Calculate advantage (similar to PerfectDou's DouDizhu advantage calculation)
        # In Dummy, we can adapt this as difference in progress towards winning
        current_advantage = self._calculate_advantage(min_steps, current_player, game)
        
        # Base reward is the advantage itself (normalized)
        reward = current_advantage / 20.0  # Normalize by max possible steps
        
        # Add bonus rewards for specific achievements
        reward += self._calculate_bonus_rewards(current_player, game)
        
        return reward * self.reward_scaling
    
    def calculate_advantage_change(self, game_before: DummyGame, game_after: DummyGame, 
                                 current_player: Player) -> float:
        """
        Calculate the change in advantage between two game states.
        This implements the core PerfectDou reward formula.
        
        Args:
            game_before: Game state before action
            game_after: Game state after action
            current_player: Player who took the action
            
        Returns:
            Advantage change reward
        """
        # Calculate advantage before action
        min_steps_before = {}
        for player in game_before.players:
            min_steps_before[player.id] = self._calculate_min_steps_to_win(player, game_before)
        advantage_before = self._calculate_advantage(min_steps_before, current_player, game_before)
        
        # Calculate advantage after action
        min_steps_after = {}
        for player in game_after.players:
            min_steps_after[player.id] = self._calculate_min_steps_to_win(player, game_after)
        advantage_after = self._calculate_advantage(min_steps_after, current_player, game_after)
        
        # Calculate advantage change
        advantage_change = advantage_after - advantage_before
        
        # Apply PerfectDou-style reward scaling
        # Positive advantage change = good for current player
        reward = advantage_change * self.reward_scaling
        
        return reward
    
    def _calculate_min_steps_to_win(self, player: Player, game: DummyGame) -> int:
        """
        Calculate minimum number of steps for a player to win.
        
        This uses dynamic programming similar to PerfectDou's oracle.
        In Dummy, winning means playing all cards through melds and final discard.
        """
        # Create state key for memoization
        state_key = self._create_state_key(player, game)
        
        if state_key in self.memo:
            return self.memo[state_key]
        
        # Base case: if player has no cards, they've won
        if len(player.cards) == 0:
            self.memo[state_key] = 0
            return 0
        
        # If player has 1 card, they need 1 step to knock/discard
        if len(player.cards) == 1:
            self.memo[state_key] = 1
            return 1
        
        # Calculate minimum steps using dynamic programming
        min_steps = self._dp_min_steps(player, game)
        
        self.memo[state_key] = min_steps
        return min_steps
    
    def _dp_min_steps(self, player: Player, game: DummyGame) -> int:
        """
        Dynamic programming calculation for minimum steps.
        
        Consider all possible actions:
        1. Form melds with current cards
        2. Layoff cards to existing melds
        3. Discard cards optimally
        """
        hand_cards = list(player.cards.cards)
        num_cards = len(hand_cards)
        
        if num_cards <= 1:
            return num_cards
        
        # Find all possible melds from current hand
        CARDS_MAP.clear()
        CARDS_MAP.update(hand_cards)
        possible_melds = CARDS_MAP.get_all_forms()
        
        min_steps = float('inf')
        
        # Option 1: Try forming each possible meld
        for meld in possible_melds:
            if len(meld.cards) >= 3:  # Valid meld size
                remaining_cards = [c for c in hand_cards if c not in meld.cards]
                
                # Recursive calculation for remaining cards
                steps_for_remaining = self._estimate_steps_for_cards(remaining_cards, game)
                total_steps = 1 + steps_for_remaining  # 1 step to form meld + steps for remaining
                
                min_steps = min(min_steps, total_steps)
        
        # Option 2: Try laying off cards to existing melds
        if game.forms:  # If there are existing melds
            for card in hand_cards:
                for existing_meld in game.forms:
                    if existing_meld.can_extend({card}):
                        remaining_cards = [c for c in hand_cards if c != card]
                        steps_for_remaining = self._estimate_steps_for_cards(remaining_cards, game)
                        total_steps = 1 + steps_for_remaining  # 1 step to layoff + steps for remaining
                        
                        min_steps = min(min_steps, total_steps)
        
        # Option 3: Optimal discard strategy (fallback)
        if min_steps == float('inf'):
            # Estimate based on card values and meld potential
            min_steps = self._estimate_discard_strategy(hand_cards, game)
        
        return int(min_steps)
    
    def _estimate_steps_for_cards(self, cards: List[Card], game: DummyGame) -> int:
        """
        Estimate minimum steps needed for a given set of cards.
        """
        if not cards:
            return 0
        
        if len(cards) == 1:
            return 1  # One discard/knock
        
        # Simple heuristic: group cards by potential melds
        CARDS_MAP.clear()
        CARDS_MAP.update(cards)
        potential_melds = CARDS_MAP.get_all_forms()
        
        if potential_melds:
            # Can form at least one meld
            best_meld = max(potential_melds, key=lambda m: len(m.cards))
            remaining = [c for c in cards if c not in best_meld.cards]
            return 1 + self._estimate_steps_for_cards(remaining, game)
        else:
            # No melds possible, need to discard optimally
            return len(cards)
    
    def _estimate_discard_strategy(self, cards: List[Card], game: DummyGame) -> int:
        """
        Estimate steps needed with optimal discard strategy.
        """
        # Simplified heuristic based on card values and meld potential
        total_cards = len(cards)
        
        # Estimate how many cards can be grouped into melds
        potential_meld_cards = 0
        CARDS_MAP.clear()
        CARDS_MAP.update(cards)
        
        # Count cards that could potentially form melds
        for rank in range(2, 15):  # All ranks
            for suit in range(4):  # All suits
                if CARDS_MAP.has_card(rank, suit):
                    # Check for potential runs
                    run_length = 0
                    for r in range(rank, min(rank + 10, 15)):
                        if CARDS_MAP.has_card(r, suit):
                            run_length += 1
                        else:
                            break
                    
                    if run_length >= 3:
                        potential_meld_cards += min(run_length, 3)
                    
                    # Check for potential sets
                    set_size = 0
                    for s in range(4):
                        if CARDS_MAP.has_card(rank, s):
                            set_size += 1
                    
                    if set_size >= 3:
                        potential_meld_cards += min(set_size, 3)
        
        # Estimate melds that can be formed
        estimated_melds = min(potential_meld_cards // 3, total_cards // 3)
        remaining_cards = total_cards - (estimated_melds * 3)
        
        # Total steps = meld formation + remaining discards
        return estimated_melds + remaining_cards
    
    def _calculate_advantage(self, min_steps: Dict[PlayerIndex, int], 
                           current_player: Player, game: DummyGame) -> float:
        """
        Calculate player's advantage in the current game state.
        
        Adapted from PerfectDou's advantage calculation for Dummy game.
        """
        current_steps = min_steps[current_player.id]
        
        # In Dummy, calculate advantage relative to other players
        other_players_steps = [min_steps[p.id] for p in game.players if p != current_player]
        
        if not other_players_steps:
            return 0.0
        
        # Advantage = how much closer to winning compared to best opponent
        best_opponent_steps = min(other_players_steps)
        advantage = best_opponent_steps - current_steps
        
        return float(advantage)
    
    def _calculate_bonus_rewards(self, player: Player, game: DummyGame) -> float:
        """
        Calculate bonus rewards for specific achievements.
        """
        bonus = 0.0
        
        # Bonus for forming melds
        if player.dropped_forms:
            bonus += len(player.dropped_forms) * 0.1
        
        # Bonus for having special cards (speto cards)
        if player.cards.has_speto():
            bonus += 0.2
        
        # Penalty for high-value cards in hand (encourages playing them)
        hand_value = sum(card.get_value() for card in player.cards.cards)
        bonus -= hand_value / 1000.0  # Small penalty for holding high-value cards
        
        # Bonus for being close to winning
        if len(player.cards) <= 3:
            bonus += 0.5
        
        return bonus
    
    def _create_state_key(self, player: Player, game: DummyGame) -> str:
        """
        Create a unique key for the current state for memoization.
        """
        # Simplified state key (in practice, this could be more sophisticated)
        hand_cards = sorted([card.id for card in player.cards.cards])
        melded_cards = sorted([card.id for form in game.forms for card in form.cards])
        discard_top = game.discard_pile[-1].id if game.discard_pile else -1
        
        return f"{player.id.value}_{hand_cards}_{melded_cards}_{discard_top}_{game.current_turn}"
    
    def clear_memo(self):
        """Clear memoization cache."""
        self.memo.clear()