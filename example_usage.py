#!/usr/bin/env python3
"""
Example usage of PTIE Dummy implementation.
This script demonstrates how to use the PTIE framework for training and evaluation.
"""

import os
import torch
import argparse
from datetime import datetime

# Import PTIE components
from ptie_dummy.training_pipeline import PTIETrainingPipeline, TrainingConfig
from ptie_dummy.evaluation import PTIEEvaluator, EvaluationConfig, OpponentType
from ptie_dummy.demo import DummyGameDemo

# Import game components
from game.game import DummyGame, TurnDirection


def example_quick_training():
    """Example: Quick training session (small scale for demonstration)."""
    print("="*60)
    print("EXAMPLE: Quick Training Session")
    print("="*60)
    
    # Setup training configuration
    config = TrainingConfig()
    config.total_timesteps = 50_000  # Small scale for demo
    config.num_envs = 4              # Fewer environments
    config.eval_frequency = 10_000   # Evaluate more frequently
    config.save_frequency = 20_000   # Save more frequently
    config.device = 'cuda'           # Use GPU for compatibility
    
    # Override model directory for example
    config.model_dir = 'example_models'
    config.log_dir = 'example_logs'
    
    print(f"Training configuration:")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Environments: {config.num_envs}")
    print(f"  Device: {config.device}")
    print(f"  Model directory: {config.model_dir}")
    
    # Create and run training pipeline
    print(f"\nStarting training...")
    pipeline = PTIETrainingPipeline(config)
    
    # Run training (this will take a while in real usage)
    try:
        pipeline.train()
        print(f"\nTraining completed successfully!")
        print(f"Models saved in: {config.model_dir}")
        
        return os.path.join(config.model_dir, 'best_model.pt')
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user.")
        return None
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        return None


def example_evaluation(model_path: str):
    """Example: Evaluate trained model."""
    print("\n" + "="*60)
    print("EXAMPLE: Model Evaluation")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Skipping evaluation example.")
        return
    
    # Setup evaluation configuration
    config = EvaluationConfig()
    config.model_path = model_path
    config.num_episodes = 100        # Smaller number for demo
    config.device = 'cpu'
    config.opponent_types = [OpponentType.RANDOM, OpponentType.GREEDY]
    config.output_dir = 'example_eval_results'
    
    print(f"Evaluation configuration:")
    print(f"  Model: {config.model_path}")
    print(f"  Episodes: {config.num_episodes}")
    print(f"  Opponents: {config.opponent_types}")
    print(f"  Device: {config.device}")
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    evaluator = PTIEEvaluator(config)
    
    try:
        results = evaluator.evaluate_comprehensive()
        
        print(f"\nEvaluation completed!")
        print(f"Results saved in: {config.output_dir}")
        
        # Display summary
        print(f"\nSUMMARY:")
        for opponent_type, stats in results.items():
            print(f"{opponent_type}: {stats['win_rate']:.1%} win rate")
        
        return results
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        return None


def example_interactive_demo(model_path: str = None):
    """Example: Interactive demo."""
    print("\n" + "="*60)
    print("EXAMPLE: Interactive Demo")
    print("="*60)
    
    if model_path and not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Running demo with random AI instead.")
        model_path = None
    
    # Create demo instance
    demo = DummyGameDemo(model_path, device='cpu')
    
    print(f"Demo setup:")
    print(f"  Model: {model_path if model_path else 'Random AI'}")
    print(f"  Device: cpu")
    
    # Ask user for demo type
    print(f"\nDemo options:")
    print(f"  1. Interactive game (human vs AI)")
    print(f"  2. AI vs AI demonstration")
    print(f"  3. Skip demo")
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == '1':
        print(f"\nStarting interactive game...")
        demo.run_interactive_game()
    elif choice == '2':
        print(f"\nStarting AI vs AI demo...")
        demo.run_ai_vs_ai_demo(num_games=3)
    else:
        print(f"Skipping demo.")


def example_game_mechanics():
    """Example: Demonstrate basic game mechanics."""
    print("\n" + "="*60)
    print("EXAMPLE: Basic Game Mechanics")
    print("="*60)
    
    # Create a new game
    game = DummyGame("example_game")
    
    print("1. Setting up game...")
    game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
    game.deal_cards()
    game.save_initial_state()
    
    print(f"   Players: {len(game.players)}")
    print(f"   Stock pile: {len(game.stock_pile)} cards")
    print(f"   Discard pile: {len(game.discard_pile)} cards")
    
    # Show initial hands
    print("\n2. Initial hands:")
    for player in game.players:
        print(f"   Player {player.id.value}: {len(player.cards)} cards")
    
    # Start first turn
    game.next_turn()
    current_player = game.current_player
    
    print(f"\n3. First turn:")
    print(f"   Current player: {current_player.id.value}")
    print(f"   Phase: {game.phase.name}")
    
    # Show legal actions
    legal_actions = game.get_all_possible_action_for_player(current_player)
    print(f"   Legal actions: {len(legal_actions)}")
    for i, action in enumerate(legal_actions[:5]):  # Show first 5
        print(f"     {i+1}. {action.action_type.name}")
    if len(legal_actions) > 5:
        print(f"     ... and {len(legal_actions) - 5} more")
    
    print(f"\nGame mechanics demonstration completed.")


def main():
    """Main example script."""
    parser = argparse.ArgumentParser(description='PTIE Dummy Examples')
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'eval', 'demo', 'mechanics', 'all'],
                       default='all',
                       help='Which example to run')
    parser.add_argument('--model', type=str,
                       help='Path to trained model (for eval/demo)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("PTIE DUMMY - EXAMPLE USAGE")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model_path = args.model
    
    if args.mode in ['train', 'all']:
        trained_model = example_quick_training()
        if trained_model and not model_path:
            model_path = trained_model
    
    if args.mode in ['eval', 'all']:
        if model_path:
            example_evaluation(model_path)
        else:
            print(f"\nSkipping evaluation - no model available")
    
    if args.mode in ['demo', 'all']:
        example_interactive_demo(model_path)
    
    if args.mode in ['mechanics', 'all']:
        example_game_mechanics()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED")
    print("="*60)
    print(f"For more advanced usage, see:")
    print(f"  - Training: python -m ptie_dummy.training_pipeline --help")
    print(f"  - Evaluation: python -m ptie_dummy.evaluation --help")
    print(f"  - Demo: python -m ptie_dummy.demo --help")


if __name__ == '__main__':
    main()