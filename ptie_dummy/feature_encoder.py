"""
Feature Encoder for Dummy Game PTIE Implementation
Handles both perfect and imperfect information encoding for the actor-critic networks.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from enum import Enum

from game.game import DummyGame, GameState, GamePhase
from game.player import Player, PlayerIndex  
from game.card import Card, CardList, Suit, Rank
from game.action import Action, ActionType
from game.form import Form


class FeatureType(Enum):
    IMPERFECT = "imperfect"  # For policy network (actor)
    PERFECT = "perfect"      # For value network (critic)


class DummyFeatureEncoder:
    """
    Encodes game state features for PTIE networks.
    
    Based on PerfectDou's feature encoding:
    - Imperfect features: player's hand, public info, action history
    - Perfect features: all players' hands, minimum steps to win
    """
    
    def __init__(self):
        self.num_cards = 52
        self.num_suits = 4 
        self.num_ranks = 13
        self.max_players = 4
        self.max_actions_history = 15
        
        # Feature dimensions
        self.imperfect_card_features = (15, 13, 4)  # history_len x ranks x suits
        self.imperfect_game_state_features = 8
        self.perfect_card_features = (17, 13, 4)    # extended with all players' hands
        self.perfect_game_state_features = 12
        
    def encode_game_state(self, game: DummyGame, current_player: Player, 
                         feature_type: FeatureType) -> Dict[str, np.ndarray]:
        """
        Encode game state for the specified feature type.
        
        Args:
            game: Current game instance
            current_player: Player for whom to encode features
            feature_type: IMPERFECT for policy network, PERFECT for value network
            
        Returns:
            Dictionary with encoded features
        """
        if feature_type == FeatureType.IMPERFECT:
            return self._encode_imperfect_features(game, current_player)
        else:
            return self._encode_perfect_features(game, current_player)
    
    def _encode_imperfect_features(self, game: DummyGame, current_player: Player) -> Dict[str, np.ndarray]:
        """Encode imperfect information features for policy network."""
        
        # Card features matrix: (history_len, ranks, suits)
        card_features = np.zeros(self.imperfect_card_features, dtype=np.float32)
        
        # Layer 0: Current player's hand
        self._encode_player_hand(card_features[0], current_player)
        
        # Layer 1: All melded forms (public information)
        self._encode_melded_forms(card_features[1], game.forms)
        
        # Layer 2: Discard pile
        self._encode_card_list(card_features[2], game.discard_pile)
        
        # Layer 3-4: Remaining cards estimation (cards not seen)
        remaining_cards = self._get_remaining_cards(game, current_player)
        self._encode_card_list(card_features[3], remaining_cards)
        
        # Layers 5-14: Last 10 actions history
        self._encode_action_history(card_features[5:15], game, current_player)
        
        # Game state features
        game_state = np.zeros(self.imperfect_game_state_features, dtype=np.float32)
        game_state[0] = len(current_player.cards) / 20.0  # Normalized hand size
        game_state[1] = len(game.discard_pile) / 52.0     # Normalized discard pile size
        game_state[2] = len(game.stock_pile) / 52.0       # Normalized stock pile size
        game_state[3] = game.current_turn / 100.0         # Normalized turn number
        game_state[4] = 1.0 if game.phase == GamePhase.PICK_OR_DRAW else 0.0
        game_state[5] = 1.0 if current_player.can_show_speto else 0.0
        game_state[6] = len(current_player.dropped_forms) / 10.0  # Normalized meld count
        game_state[7] = current_player.score / 500.0      # Normalized current score
        
        return {
            'card_features': card_features.flatten(),
            'game_state': game_state
        }
    
    def _encode_perfect_features(self, game: DummyGame, current_player: Player) -> Dict[str, np.ndarray]:
        """Encode perfect information features for value network."""
        
        # Extended card features matrix: (history_len, ranks, suits)
        card_features = np.zeros(self.perfect_card_features, dtype=np.float32)
        
        # Copy imperfect features first
        imperfect_features = self._encode_imperfect_features(game, current_player)
        card_features[:15] = imperfect_features['card_features'].reshape(self.imperfect_card_features)
        
        # Layer 15: All other players' hands (perfect information)
        combined_other_hands = []
        for player in game.players:
            if player != current_player:
                combined_other_hands.extend(player.cards.cards)
        self._encode_card_list(card_features[15], combined_other_hands)
        
        # Layer 16: Minimum steps calculation for each player
        self._encode_min_steps_layer(card_features[16], game)
        
        # Extended game state features
        game_state = np.zeros(self.perfect_game_state_features, dtype=np.float32)
        
        # Copy imperfect game state
        game_state[:8] = imperfect_features['game_state']
        
        # Add perfect information features
        for i, player in enumerate(game.players):
            if i < 4:  # Support up to 4 players
                game_state[8 + i] = self._calculate_min_steps_to_win(player, game) / 20.0
        
        return {
            'card_features': card_features.flatten(),
            'game_state': game_state
        }
    
    def encode_legal_actions(self, game: DummyGame, current_player: Player, max_actions: int = 20) -> Tuple[np.ndarray, List]:
        """
        Encode legal actions for the current player with padding.
        
        Args:
            max_actions: Maximum number of actions to pad to
            
        Returns:
            Tuple of (action_features, legal_actions_list)
        """
        legal_actions = game.get_all_possible_action_for_player(current_player)
        action_features = []
        
        for action in legal_actions:
            action_feature = self._encode_single_action(action, game, current_player)
            action_features.append(action_feature)
        
        if not action_features:
            # No legal actions (shouldn't happen in normal play)
            action_features = [np.zeros(100, dtype=np.float32)]
        
        # Pad to max_actions
        while len(action_features) < max_actions:
            action_features.append(np.zeros(100, dtype=np.float32))
        
        # Truncate if too many actions
        action_features = action_features[:max_actions]
        
        # Convert to array with consistent shape
        action_features_array = np.array(action_features, dtype=np.float32)
        
        return action_features_array, legal_actions
    
    def _encode_single_action(self, action: Action, game: DummyGame, current_player: Player) -> np.ndarray:
        """Encode a single action into a feature vector."""
        feature = np.zeros(100, dtype=np.float32)  # Action feature vector size
        
        # Action type encoding (one-hot)
        action_type_map = {
            ActionType.DRAW: 0,
            ActionType.PICK: 1, 
            ActionType.MELD: 2,
            ActionType.LAYOFF: 3,
            ActionType.DISCARD: 4,
            ActionType.SHOW_SPETO: 5,
            ActionType.KNOCK: 6
        }
        
        if action.action_type in action_type_map:
            feature[action_type_map[action.action_type]] = 1.0
        
        # Encode action-specific information
        if hasattr(action, 'card') and action.card:
            # Encode card information
            feature[10 + action.card.rank.value] = 1.0      # Rank
            feature[25 + action.card.suit.value] = 1.0      # Suit
            feature[30] = action.card.get_value() / 50.0    # Normalized card value
        
        if hasattr(action, 'form') and action.form:
            # Encode form information
            feature[40] = len(action.form.cards) / 13.0     # Form size
            feature[41] = 1.0 if action.form.is_set else 0.0
            feature[42] = 1.0 if action.form.is_run else 0.0
            feature[43] = sum(card.get_value() for card in action.form.cards) / 200.0  # Form value
        
        return feature
    
    def _encode_player_hand(self, layer: np.ndarray, player: Player):
        """Encode player's hand cards into the feature layer."""
        for card in player.cards:
            layer[card.rank.value - 2, card.suit.value] = 1.0
    
    def _encode_card_list(self, layer: np.ndarray, cards: List[Card]):
        """Encode a list of cards into the feature layer."""
        for card in cards:
            layer[card.rank.value - 2, card.suit.value] = 1.0
    
    def _encode_melded_forms(self, layer: np.ndarray, forms: List[Form]):
        """Encode all melded forms into the feature layer."""
        for form in forms:
            for card in form.cards:
                layer[card.rank.value - 2, card.suit.value] = 1.0
    
    def _get_remaining_cards(self, game: DummyGame, current_player: Player) -> List[Card]:
        """Get cards not visible to current player."""
        all_cards = set([Card.get_card_from_id(i) for i in range(2, 54)])
        visible_cards = set(current_player.cards.cards)
        visible_cards.update(game.discard_pile)
        
        # Add cards from melded forms
        for form in game.forms:
            visible_cards.update(form.cards)
        
        return list(all_cards - visible_cards)
    
    def _encode_action_history(self, layers: np.ndarray, game: DummyGame, current_player: Player):
        """Encode recent action history."""
        # This would need access to game action history
        # For now, we'll leave this as zeros (can be enhanced later)
        pass
    
    def _encode_min_steps_layer(self, layer: np.ndarray, game: DummyGame):
        """Encode minimum steps to win for each position."""
        for i, player in enumerate(game.players):
            min_steps = self._calculate_min_steps_to_win(player, game)
            # Encode this information spatially (simplified)
            if i < layer.shape[0]:
                for j in range(layer.shape[1]):
                    layer[i, j] = min_steps / 20.0  # Normalized
    
    def _calculate_min_steps_to_win(self, player: Player, game: DummyGame) -> int:
        """
        Calculate minimum steps for player to win (oracle function).
        
        This is a simplified version - in PerfectDou this uses dynamic programming.
        """
        # Simplified calculation based on cards in hand and possible melds
        hand_cards = len(player.cards)
        melded_cards = sum(len(form.cards) for form in player.dropped_forms)
        
        # Estimate minimum moves needed
        remaining_cards = hand_cards
        estimated_moves = 0
        
        # Need to form melds with remaining cards
        while remaining_cards > 1:
            if remaining_cards >= 3:
                # Can form a meld
                remaining_cards -= 3
                estimated_moves += 1
            else:
                # Need to discard
                remaining_cards -= 1
                estimated_moves += 1
        
        # Final knock
        estimated_moves += 1
        
        return estimated_moves

    def get_feature_dimensions(self, feature_type: FeatureType) -> Tuple[int, int]:
        """
        Get feature dimensions for the specified feature type.
        
        Returns:
            Tuple of (card_features_size, game_state_size)
        """
        if feature_type == FeatureType.IMPERFECT:
            card_size = np.prod(self.imperfect_card_features)
            return card_size, self.imperfect_game_state_features
        else:
            card_size = np.prod(self.perfect_card_features)
            return card_size, self.perfect_game_state_features