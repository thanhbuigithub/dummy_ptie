"""
Demo script for PTIE Dummy implementation.
Shows how to use the trained AI to play against human or other opponents.
"""

import torch
import argparse
import os
from typing import Optional

from .networks import PTIEActorCritic
from .feature_encoder import DummyFeatureEncoder, FeatureType
from game.game import DummyGame, TurnDirection, GameState
from game.player import PlayerIndex
from game.action import ActionType


class DummyGameDemo:
    """Interactive demo for PTIE Dummy AI."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.feature_encoder = DummyFeatureEncoder()
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.model = self._load_model(model_path)
            print(f"Loaded trained model from {model_path}")
        else:
            print("No model provided. AI will play randomly.")
            self.model = None
    
    def run_interactive_game(self):
        """Run an interactive game where human plays against AI."""
        print("\n" + "="*60)
        print("Welcome to PTIE Dummy Game Demo!")
        print("="*60)
        print("You are Player 0. The AI will control other players.")
        print("Available actions will be numbered for easy selection.")
        print("="*60)
        
        game = DummyGame("demo_game")
        game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
        game.deal_cards()
        game.save_initial_state()
        game.next_turn()
        
        while game.game_state == GameState.STARTED:
            current_player = game.current_player
            
            # Display game state
            self._display_game_state(game)
            
            if current_player.id == PlayerIndex.PLAYER_0:
                # Human player
                action = self._get_human_action(game, current_player)
            else:
                # AI player
                action = self._get_ai_action(game, current_player)
                if action:
                    print(f"\nPlayer {current_player.id.value} (AI) chose: {action}")
            
            if action:
                try:
                    game.take_action(action)
                except Exception as e:
                    print(f"Error executing action: {e}")
                    break
            else:
                print("No valid action available!")
                break
            
            input("\nPress Enter to continue...")
        
        # Game finished
        self._display_final_results(game)
    
    def run_ai_vs_ai_demo(self, num_games: int = 5):
        """Run demo with AI vs AI games."""
        print(f"\n{'='*60}")
        print(f"AI vs AI Demo - {num_games} games")
        print(f"{'='*60}")
        
        wins = {PlayerIndex.PLAYER_0: 0, PlayerIndex.PLAYER_1: 0, 
                PlayerIndex.PLAYER_2: 0, PlayerIndex.PLAYER_3: 0}
        
        for game_num in range(num_games):
            print(f"\nGame {game_num + 1}/{num_games}")
            print("-" * 30)
            
            game = DummyGame(f"demo_game_{game_num}")
            game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
            game.deal_cards()
            game.save_initial_state()
            game.next_turn()
            
            turn_count = 0
            max_turns = 200
            
            while game.game_state == GameState.STARTED and turn_count < max_turns:
                current_player = game.current_player
                action = self._get_ai_action(game, current_player)
                
                if action:
                    try:
                        print(f"Player {current_player.id.value}: {action}")
                        game.take_action(action)
                        turn_count += 1
                    except Exception as e:
                        print(f"Error: {e}")
                        break
                else:
                    break
            
            # Display results
            if game.game_state == GameState.FINISHED:
                final_scores = {player.id: player.calculate_end_score() for player in game.players}
                winner = min(final_scores.keys(), key=lambda x: final_scores[x])
                wins[winner] += 1
                
                print(f"Winner: Player {winner.value}")
                print("Final scores:")
                for player_id, score in final_scores.items():
                    print(f"  Player {player_id.value}: {score}")
            else:
                print("Game ended without completion")
        
        # Summary
        print(f"\n{'='*60}")
        print("DEMO SUMMARY")
        print(f"{'='*60}")
        for player_id, win_count in wins.items():
            win_rate = win_count / num_games * 100
            print(f"Player {player_id.value}: {win_count}/{num_games} wins ({win_rate:.1f}%)")
    
    def _display_game_state(self, game: DummyGame):
        """Display current game state."""
        print(f"\n{'='*60}")
        print(f"Turn {game.current_turn} - Player {game.current_player.id.value}'s turn")
        print(f"Phase: {game.phase.name}")
        print(f"{'='*60}")
        
        # Show current player's hand
        current_player = game.current_player
        print(f"\nYour hand ({len(current_player.cards)} cards):")
        for i, card in enumerate(sorted(current_player.cards.cards, key=lambda c: c.id)):
            print(f"  {i+1}. {card}")
        
        # Show discard pile
        if game.discard_pile:
            print(f"\nTop of discard pile: {game.discard_pile[-1]}")
        
        # Show melded forms
        if game.forms:
            print(f"\nMelded forms on table:")
            for i, form in enumerate(game.forms):
                owner_name = f"Player {form.owner}"
                form_type = "Set" if form.is_set else "Run" if form.is_run else "Unknown"
                cards_str = ", ".join(str(card) for card in sorted(form.cards, key=lambda c: c.id))
                print(f"  {i+1}. {owner_name} ({form_type}): {cards_str}")
        
        # Show player scores
        print(f"\nCurrent scores:")
        for player in game.players:
            print(f"  Player {player.id.value}: {player.score} points")
    
    def _get_human_action(self, game: DummyGame, player):
        """Get action from human player."""
        legal_actions = game.get_all_possible_action_for_player(player)
        
        if not legal_actions:
            print("No legal actions available!")
            return None
        
        print(f"\nAvailable actions:")
        for i, action in enumerate(legal_actions):
            action_desc = self._describe_action(action)
            print(f"  {i+1}. {action_desc}")
        
        while True:
            try:
                choice = input(f"\nSelect action (1-{len(legal_actions)}): ").strip()
                if choice.lower() in ['q', 'quit']:
                    return None
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(legal_actions):
                    return legal_actions[choice_idx]
                else:
                    print(f"Invalid choice. Please enter 1-{len(legal_actions)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nGame terminated by user.")
                return None
    
    def _get_ai_action(self, game: DummyGame, player):
        """Get action from AI player."""
        if self.model is None:
            # Random action
            legal_actions = game.get_all_possible_action_for_player(player)
            import random
            return random.choice(legal_actions) if legal_actions else None
        
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
                    'card_features': torch.FloatTensor(imperfect_features['card_features']).unsqueeze(0).to(self.device),
                    'game_state': torch.FloatTensor(imperfect_features['game_state']).unsqueeze(0).to(self.device)
                }
                legal_actions_tensor = torch.FloatTensor(legal_actions).unsqueeze(0).to(self.device)
                
                # Get action
                action_dist = self.model.get_action_distribution(imperfect_tensors, legal_actions_tensor)
                action_idx = action_dist.sample().item()
                
                if action_idx < len(legal_actions_list):
                    return legal_actions_list[action_idx]
                else:
                    import random
                    return random.choice(legal_actions_list) if legal_actions_list else None
                    
        except Exception as e:
            print(f"AI action selection failed: {e}")
            legal_actions_list = game.get_all_possible_action_for_player(player)
            import random
            return random.choice(legal_actions_list) if legal_actions_list else None
    
    def _describe_action(self, action):
        """Create human-readable description of action."""
        if action.action_type == ActionType.DRAW:
            return "Draw a card from stock pile"
        elif action.action_type == ActionType.PICK:
            return f"Pick {action.card} and meld {len(action.form.cards)} cards"
        elif action.action_type == ActionType.MELD:
            cards_str = ", ".join(str(card) for card in action.form.cards)
            form_type = "set" if action.form.is_set else "run"
            return f"Meld {form_type}: {cards_str}"
        elif action.action_type == ActionType.LAYOFF:
            return f"Layoff {action.layoff.cards.cards} to existing meld"
        elif action.action_type == ActionType.DISCARD:
            return f"Discard {action.card}"
        elif action.action_type == ActionType.KNOCK:
            return f"Knock with {action.card}"
        elif action.action_type == ActionType.SHOW_SPETO:
            return "Show SPETO (2♣ and Q♠)"
        else:
            return str(action)
    
    def _display_final_results(self, game: DummyGame):
        """Display final game results."""
        print(f"\n{'='*60}")
        print("GAME FINISHED!")
        print(f"{'='*60}")
        
        if game.game_state == GameState.FINISHED:
            final_scores = {}
            for player in game.players:
                final_score = player.calculate_end_score()
                final_scores[player.id] = final_score
                print(f"Player {player.id.value} final score: {final_score}")
                
                # Show extra points
                if player.extra_points:
                    print(f"  Extra points:")
                    for extra in player.extra_points:
                        print(f"    {extra.type.name}: {extra.value}")
            
            # Determine winner (lowest score)
            winner = min(final_scores.keys(), key=lambda x: final_scores[x])
            print(f"\nWINNER: Player {winner.value} with {final_scores[winner]} points!")
        else:
            print("Game ended without completion.")
    
    def _load_model(self, model_path: str):
        """Load trained model."""
        # Get feature dimensions
        imperfect_card_dim, imperfect_state_dim = self.feature_encoder.get_feature_dimensions(FeatureType.IMPERFECT)
        perfect_card_dim, perfect_state_dim = self.feature_encoder.get_feature_dimensions(FeatureType.PERFECT)
        
        # Create model
        model = PTIEActorCritic(
            imperfect_card_dim, imperfect_state_dim,
            perfect_card_dim, perfect_state_dim
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


def main():
    """Main demo script."""
    parser = argparse.ArgumentParser(description='PTIE Dummy Game Demo')
    parser.add_argument('--model', type=str, help='Path to trained model (optional)')
    parser.add_argument('--mode', type=str, choices=['interactive', 'ai_vs_ai'], 
                       default='interactive', help='Demo mode')
    parser.add_argument('--games', type=int, default=5, help='Number of games for AI vs AI mode')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    
    demo = DummyGameDemo(args.model, device)
    
    if args.mode == 'interactive':
        demo.run_interactive_game()
    elif args.mode == 'ai_vs_ai':
        demo.run_ai_vs_ai_demo(args.games)


if __name__ == '__main__':
    main()