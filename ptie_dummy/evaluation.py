"""
Evaluation script for PTIE Dummy implementation.
Provides comprehensive evaluation against different opponents.
"""

import torch
import numpy as np
import random
import argparse
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime

from .training_pipeline import PTIETrainingPipeline, TrainingConfig
from .networks import PTIEActorCritic
from .feature_encoder import DummyFeatureEncoder, FeatureType
from game.game import DummyGame, TurnDirection
from game.player import PlayerIndex
from game.action import ActionType


class OpponentType:
    """Types of opponents for evaluation."""
    RANDOM = "random"
    GREEDY = "greedy"
    RULE_BASED = "rule_based"


class EvaluationConfig:
    """Configuration for evaluation."""
    
    def __init__(self):
        self.num_episodes = 1000
        self.model_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.opponent_types = [OpponentType.RANDOM, OpponentType.GREEDY]
        self.output_dir = 'eval_results'
        self.verbose = True


class PTIEEvaluator:
    """Comprehensive evaluation for PTIE Dummy AI."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.feature_encoder = DummyFeatureEncoder()
        
        # Load trained model
        if config.model_path and os.path.exists(config.model_path):
            self.model = self._load_model(config.model_path)
            print(f"Loaded model from {config.model_path}")
        else:
            print("Warning: No model loaded. Using random policy.")
            self.model = None
        
        os.makedirs(config.output_dir, exist_ok=True)
    
    def evaluate_comprehensive(self) -> Dict[str, Dict[str, float]]:
        """
        Run comprehensive evaluation against all opponent types.
        
        Returns:
            Dictionary with results for each opponent type
        """
        results = {}
        
        for opponent_type in self.config.opponent_types:
            print(f"\nEvaluating against {opponent_type} opponents...")
            opponent_results = self.evaluate_vs_opponent(opponent_type, self.config.num_episodes)
            results[opponent_type] = opponent_results
            
            if self.config.verbose:
                self._print_results(opponent_type, opponent_results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def evaluate_vs_opponent(self, opponent_type: str, num_episodes: int) -> Dict[str, float]:
        """
        Evaluate against a specific opponent type.
        
        Args:
            opponent_type: Type of opponent
            num_episodes: Number of episodes to run
            
        Returns:
            Dictionary with evaluation statistics
        """
        wins = 0
        losses = 0
        draws = 0
        episode_lengths = []
        scores = []
        ai_final_scores = []
        
        for episode in range(num_episodes):
            result = self._run_single_game(opponent_type, episode)
            
            if result['winner'] == PlayerIndex.PLAYER_0:
                wins += 1
            elif result['winner'] is None:
                draws += 1
            else:
                losses += 1
            
            episode_lengths.append(result['episode_length'])
            scores.append(result['ai_score'])
            ai_final_scores.append(result['ai_final_score'])
        
        total_games = wins + losses + draws
        
        return {
            'win_rate': wins / total_games if total_games > 0 else 0.0,
            'loss_rate': losses / total_games if total_games > 0 else 0.0,
            'draw_rate': draws / total_games if total_games > 0 else 0.0,
            'avg_episode_length': np.mean(episode_lengths),
            'avg_score': np.mean(scores),
            'avg_final_score': np.mean(ai_final_scores),
            'total_games': total_games
        }
    
    def _run_single_game(self, opponent_type: str, episode_id: int) -> Dict:
        """Run a single game and return results."""
        game = DummyGame(f"eval_{episode_id}")
        game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
        game.deal_cards()
        game.save_initial_state()
        game.next_turn()
        
        episode_length = 0
        max_steps = 300  # Prevent infinite games
        ai_score = 0
        
        while game.game_state.name == 'STARTED' and episode_length < max_steps:
            current_player = game.current_player
            
            # Select action based on player type
            if current_player.id == PlayerIndex.PLAYER_0:
                # AI player
                action = self._select_ai_action(game, current_player)
            else:
                # Opponent
                action = self._select_opponent_action(game, current_player, opponent_type)
            
            if action:
                try:
                    game.take_action(action)
                    episode_length += 1
                    
                    # Track AI score
                    if current_player.id == PlayerIndex.PLAYER_0:
                        ai_score = current_player.score
                        
                except Exception as e:
                    print(f"Action execution failed: {e}")
                    break
            else:
                break
        
        # Determine winner
        winner = None
        ai_final_score = 0
        
        if game.game_state.name == 'FINISHED':
            # Winner is player with lowest final score
            final_scores = {player.id: player.calculate_end_score() for player in game.players}
            winner = min(final_scores.keys(), key=lambda x: final_scores[x])
            ai_final_score = final_scores.get(PlayerIndex.PLAYER_0, 0)
        
        return {
            'winner': winner,
            'episode_length': episode_length,
            'ai_score': ai_score,
            'ai_final_score': ai_final_score
        }
    
    def _select_ai_action(self, game: DummyGame, player):
        """Select action using AI model."""
        if self.model is None:
            return self._select_random_action(game, player)
        
        try:
            self.model.eval()
            with torch.no_grad():
                # Encode features
                imperfect_features = self.feature_encoder.encode_game_state(
                    game, player, FeatureType.IMPERFECT
                )
                legal_actions, legal_actions_list = self.feature_encoder.encode_legal_actions(game, player)
                
                # Convert to tensors
                imperfect_tensors = {
                    'card_features': torch.FloatTensor(imperfect_features['card_features']).unsqueeze(0).to(self.config.device),
                    'game_state': torch.FloatTensor(imperfect_features['game_state']).unsqueeze(0).to(self.config.device)
                }
                legal_actions_tensor = torch.FloatTensor(legal_actions).unsqueeze(0).to(self.config.device)
                
                # Get action distribution
                action_dist = self.model.get_action_distribution(imperfect_tensors, legal_actions_tensor)
                action_idx = action_dist.sample().item()
                
                # Get actual action
                if action_idx < len(legal_actions_list):
                    return legal_actions_list[action_idx]
                else:
                    return self._select_random_action(game, player)
                    
        except Exception as e:
            print(f"AI action selection failed: {e}")
            return self._select_random_action(game, player)
    
    def _select_opponent_action(self, game: DummyGame, player, opponent_type: str):
        """Select action for opponent based on type."""
        if opponent_type == OpponentType.RANDOM:
            return self._select_random_action(game, player)
        elif opponent_type == OpponentType.GREEDY:
            return self._select_greedy_action(game, player)
        elif opponent_type == OpponentType.RULE_BASED:
            return self._select_rule_based_action(game, player)
        else:
            return self._select_random_action(game, player)
    
    def _select_random_action(self, game: DummyGame, player):
        """Select random action."""
        legal_actions = game.get_all_possible_action_for_player(player)
        return random.choice(legal_actions) if legal_actions else None
    
    def _select_greedy_action(self, game: DummyGame, player):
        """Select greedy action (prioritize melds and low-value discards)."""
        legal_actions = game.get_all_possible_action_for_player(player)
        
        if not legal_actions:
            return None
        
        # Prioritize actions by type
        action_priorities = {
            ActionType.KNOCK: 5,    # Highest priority - win the game
            ActionType.MELD: 4,     # Form melds to reduce hand
            ActionType.LAYOFF: 3,   # Layoff cards to reduce hand
            ActionType.PICK: 2,     # Pick useful cards
            ActionType.SHOW_SPETO: 2,  # Show speto for bonus points
            ActionType.DRAW: 1,     # Draw new cards
            ActionType.DISCARD: 0   # Lowest priority
        }
        
        # Sort actions by priority
        sorted_actions = sorted(legal_actions, 
                               key=lambda a: action_priorities.get(a.action_type, 0), 
                               reverse=True)
        
        # Among actions of the same type, apply specific logic
        best_action = sorted_actions[0]
        
        if best_action.action_type == ActionType.DISCARD:
            # For discard, choose highest value card
            discard_actions = [a for a in legal_actions if a.action_type == ActionType.DISCARD]
            best_action = max(discard_actions, key=lambda a: a.card.get_value())
        
        return best_action
    
    def _select_rule_based_action(self, game: DummyGame, player):
        """Select action using simple rule-based strategy."""
        # For now, use greedy strategy
        # This can be enhanced with more sophisticated rules
        return self._select_greedy_action(game, player)
    
    def _load_model(self, model_path: str):
        """Load trained model."""
        # Get feature dimensions
        imperfect_card_dim, imperfect_state_dim = self.feature_encoder.get_feature_dimensions(FeatureType.IMPERFECT)
        perfect_card_dim, perfect_state_dim = self.feature_encoder.get_feature_dimensions(FeatureType.PERFECT)
        
        # Create model
        model = PTIEActorCritic(
            imperfect_card_dim, imperfect_state_dim,
            perfect_card_dim, perfect_state_dim
        ).to(self.config.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _print_results(self, opponent_type: str, results: Dict[str, float]):
        """Print evaluation results."""
        print(f"\nResults vs {opponent_type}:")
        print(f"  Win Rate: {results['win_rate']:.3f}")
        print(f"  Loss Rate: {results['loss_rate']:.3f}")
        print(f"  Draw Rate: {results['draw_rate']:.3f}")
        print(f"  Avg Episode Length: {results['avg_episode_length']:.1f}")
        print(f"  Avg Score: {results['avg_score']:.1f}")
        print(f"  Avg Final Score: {results['avg_final_score']:.1f}")
        print(f"  Total Games: {results['total_games']}")
    
    def _save_results(self, results: Dict[str, Dict[str, float]]):
        """Save evaluation results to file."""
        output_file = os.path.join(
            self.config.output_dir, 
            f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'num_episodes': self.config.num_episodes,
                    'model_path': self.config.model_path,
                    'opponent_types': self.config.opponent_types
                },
                'results': results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\nResults saved to {output_file}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate PTIE Dummy AI')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--output_dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--opponents', nargs='+', 
                       choices=[OpponentType.RANDOM, OpponentType.GREEDY, OpponentType.RULE_BASED],
                       default=[OpponentType.RANDOM, OpponentType.GREEDY],
                       help='Opponent types to evaluate against')
    
    args = parser.parse_args()
    
    # Setup config
    config = EvaluationConfig()
    config.model_path = args.model
    config.num_episodes = args.episodes
    config.output_dir = args.output_dir
    config.opponent_types = args.opponents
    
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device
    
    # Run evaluation
    evaluator = PTIEEvaluator(config)
    results = evaluator.evaluate_comprehensive()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for opponent_type, stats in results.items():
        print(f"\n{opponent_type.upper()}:")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Avg Final Score: {stats['avg_final_score']:.1f}")


if __name__ == '__main__':
    main()