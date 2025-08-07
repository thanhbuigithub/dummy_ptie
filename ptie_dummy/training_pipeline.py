"""
Training Pipeline for PTIE Dummy Implementation
Main training loop with self-play and evaluation.
"""

import torch
import numpy as np
import random
import logging
import os
from typing import List, Dict, Optional
import argparse
from datetime import datetime
import json
import time
from tqdm import tqdm

from .ppo_trainer import PPOTrainer, PPOConfig
from .networks import PTIEActorCritic
from .feature_encoder import DummyFeatureEncoder, FeatureType
from game.game import DummyGame, TurnDirection
from game.player import PlayerIndex


class TrainingConfig:
    """Configuration for the training pipeline."""
    
    def __init__(self):
        # Training parameters
        self.total_timesteps = 10_000_000
        self.num_envs = 16
        self.eval_frequency = 50_000
        self.save_frequency = 100_000
        self.log_frequency = 5_000  # More frequent for better monitoring
        
        # Model parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        
        # Directories
        self.model_dir = 'models'
        self.log_dir = 'logs'
        self.eval_dir = 'eval_results'
        
        # Self-play parameters
        self.self_play_ratio = 1.0  # Ratio of self-play vs random opponents
        
        # PPO configuration
        self.ppo_config = PPOConfig()


class PTIETrainingPipeline:
    """Main training pipeline for PTIE Dummy implementation."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Set random seeds
        self._set_seeds(self.config.seed)
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize trainer
        self.trainer = PPOTrainer(self.config.ppo_config, self.config.device)
        
        # Initialize environments
        self.training_envs = self._create_training_environments()
        self.eval_envs = self._create_evaluation_environments()
        
        # Training statistics
        self.global_step = 0
        self.episode_count = 0
        self.best_win_rate = 0.0
        
        self.logger.info(f"Initialized PTIE training pipeline on {self.config.device}")
        self.logger.info(f"Training environments: {len(self.training_envs)}")
        self.logger.info(f"Evaluation environments: {len(self.eval_envs)}")
    
    def train(self):
        """Main training loop with progress tracking."""
        self.logger.info("Starting PTIE training...")
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=self.config.total_timesteps,
            desc="Training Progress",
            unit="steps",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        last_log_time = time.time()
        training_start_time = time.time()
        
        while self.global_step < self.config.total_timesteps:
            step_start_time = time.time()
            
            # Training step
            train_stats = self.trainer.train_step(self.training_envs)
            old_global_step = self.global_step
            self.global_step = self.trainer.step_count
            
            # Update progress bar
            steps_progress = self.global_step - old_global_step
            progress_bar.update(steps_progress)
            
            # Update progress bar postfix with current stats
            current_time = time.time()
            steps_per_sec = steps_progress / (current_time - step_start_time) if (current_time - step_start_time) > 0 else 0
            
            progress_bar.set_postfix({
                'Loss': f"{train_stats['policy_loss']:.4f}",
                'Value': f"{train_stats['value_loss']:.4f}", 
                'Entropy': f"{train_stats['entropy']:.4f}",
                'Steps/s': f"{steps_per_sec:.1f}"
            })
            
            # Detailed logging (less frequent)
            if self.global_step % self.config.log_frequency == 0:
                elapsed_time = current_time - training_start_time
                self._log_training_stats(train_stats, elapsed_time)
            
            # Evaluation
            if self.global_step % self.config.eval_frequency == 0:
                progress_bar.write("Running evaluation...")
                eval_stats = self.evaluate()
                self._log_evaluation_stats(eval_stats)
                
                # Save best model
                if eval_stats['win_rate'] > self.best_win_rate:
                    self.best_win_rate = eval_stats['win_rate']
                    self._save_model('best_model.pt')
                    progress_bar.write(f"✓ New best model saved with win rate: {self.best_win_rate:.3f}")
            
            # Save checkpoint
            if self.global_step % self.config.save_frequency == 0:
                self._save_model(f'checkpoint_{self.global_step}.pt')
                progress_bar.write(f"✓ Checkpoint saved at step {self.global_step}")
            
            # Reset environments periodically  
            if self.global_step % (self.config.eval_frequency * 2) == 0:
                self._reset_training_environments()
                progress_bar.write("↻ Training environments reset")
        
        progress_bar.close()
        total_time = time.time() - training_start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds!")
        self._save_model('final_model.pt')
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the current model against random and rule-based opponents.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation statistics
        """
        self.logger.info(f"Starting evaluation for {num_episodes} episodes...")
        
        wins = 0
        total_episodes = 0
        episode_lengths = []
        rewards = []
        
        self.trainer.model.eval()
        
        with torch.no_grad():
            for episode in range(num_episodes):
                # Create fresh evaluation game
                game = DummyGame(f"eval_{episode}")
                game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
                game.deal_cards()
                game.save_initial_state()
                game.next_turn()
                
                episode_reward = 0
                episode_length = 0
                max_steps = 200  # Prevent infinite games
                
                while game.game_state.name == 'STARTED' and episode_length < max_steps:
                    current_player = game.current_player
                    
                    if current_player.id == PlayerIndex.PLAYER_0:
                        # AI player
                        action = self._select_ai_action(game, current_player)
                    else:
                        # Random opponent
                        action = self._select_random_action(game, current_player)
                    
                    if action:
                        try:
                            # Calculate reward for AI player
                            if current_player.id == PlayerIndex.PLAYER_0:
                                old_reward = self.trainer.reward_calculator.calculate_oracle_reward(game, current_player)
                            
                            game.take_action(action)
                            
                            if current_player.id == PlayerIndex.PLAYER_0:
                                new_reward = self.trainer.reward_calculator.calculate_oracle_reward(game, current_player)
                                episode_reward += new_reward - old_reward
                            
                            episode_length += 1
                            
                        except Exception as e:
                            self.logger.warning(f"Action execution failed in evaluation: {e}")
                            break
                    else:
                        break
                
                # Check if AI player won
                if game.game_state.name == 'FINISHED':
                    # Find winner (player with lowest score in Dummy)
                    scores = {player.id: player.score for player in game.players}
                    winner_id = min(scores.keys(), key=lambda x: scores[x])
                    
                    if winner_id == PlayerIndex.PLAYER_0:
                        wins += 1
                
                total_episodes += 1
                episode_lengths.append(episode_length)
                rewards.append(episode_reward)
        
        # Calculate statistics
        win_rate = wins / total_episodes if total_episodes > 0 else 0.0
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0
        avg_reward = np.mean(rewards) if rewards else 0.0
        
        eval_stats = {
            'win_rate': win_rate,
            'avg_episode_length': avg_episode_length,
            'avg_reward': avg_reward,
            'total_episodes': total_episodes
        }
        
        return eval_stats
    
    def _select_ai_action(self, game: DummyGame, player):
        """Select action using the trained AI model."""
        try:
            # Encode features
            imperfect_features = self.trainer.feature_encoder.encode_game_state(
                game, player, FeatureType.IMPERFECT
            )
            legal_actions, legal_actions_list = self.trainer.feature_encoder.encode_legal_actions(game, player)
            
            # Convert to tensors
            imperfect_tensors = {
                'card_features': torch.FloatTensor(imperfect_features['card_features']).unsqueeze(0).to(self.config.device),
                'game_state': torch.FloatTensor(imperfect_features['game_state']).unsqueeze(0).to(self.config.device)
            }
            legal_actions_tensor = torch.FloatTensor(legal_actions).unsqueeze(0).to(self.config.device)
            
            # Get action distribution
            action_dist = self.trainer.model.get_action_distribution(imperfect_tensors, legal_actions_tensor)
            action_idx = action_dist.sample().item()
            
            # Get actual action
            if action_idx < len(legal_actions_list):
                return legal_actions_list[action_idx]
            else:
                # Fallback to random action
                return random.choice(legal_actions_list) if legal_actions_list else None
                
        except Exception as e:
            self.logger.warning(f"AI action selection failed: {e}")
            # Fallback to random action
            legal_actions_list = game.get_all_possible_action_for_player(player)
            return random.choice(legal_actions_list) if legal_actions_list else None
    
    def _select_random_action(self, game: DummyGame, player):
        """Select random action for opponent."""
        legal_actions = game.get_all_possible_action_for_player(player)
        return random.choice(legal_actions) if legal_actions else None
    
    def _create_training_environments(self) -> List[DummyGame]:
        """Create training environments."""
        envs = []
        for i in range(self.config.num_envs):
            game = DummyGame(f"train_{i}")
            game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
            game.deal_cards()
            game.save_initial_state()
            game.next_turn()
            envs.append(game)
        return envs
    
    def _create_evaluation_environments(self) -> List[DummyGame]:
        """Create evaluation environments."""
        envs = []
        for i in range(4):  # Smaller number for evaluation
            game = DummyGame(f"eval_{i}")
            envs.append(game)
        return envs
    
    def _reset_training_environments(self):
        """Reset training environments."""
        for game in self.training_envs:
            game.reset()
            game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
            game.deal_cards()
            game.save_initial_state()
            game.next_turn()
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _setup_directories(self):
        """Setup necessary directories."""
        for directory in [self.config.model_dir, self.config.log_dir, self.config.eval_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.config.log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _log_training_stats(self, stats: Dict[str, float], elapsed_time: float = 0):
        """Log training statistics with performance metrics."""
        steps_per_sec = self.global_step / elapsed_time if elapsed_time > 0 else 0
        
        self.logger.info(
            f"Step {self.global_step:,} | "
            f"Policy Loss: {stats['policy_loss']:.4f} | "
            f"Value Loss: {stats['value_loss']:.4f} | "
            f"Entropy: {stats['entropy']:.4f} | "
            f"Performance: {steps_per_sec:.1f} steps/s | "
            f"Elapsed: {elapsed_time:.1f}s"
        )
    
    def _log_evaluation_stats(self, stats: Dict[str, float]):
        """Log evaluation statistics."""
        self.logger.info(
            f"Evaluation at step {self.global_step}: "
            f"Win Rate: {stats['win_rate']:.3f}, "
            f"Avg Episode Length: {stats['avg_episode_length']:.1f}, "
            f"Avg Reward: {stats['avg_reward']:.3f}"
        )
        
        # Save evaluation results
        eval_file = os.path.join(self.config.eval_dir, f'eval_{self.global_step}.json')
        with open(eval_file, 'w') as f:
            json.dump({
                'step': self.global_step,
                'stats': stats,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def _save_model(self, filename: str):
        """Save model checkpoint."""
        filepath = os.path.join(self.config.model_dir, filename)
        self.trainer.save_model(filepath)
        self.logger.info(f"Model saved to {filepath}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train PTIE Dummy AI')
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Total training timesteps')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--model_dir', type=str, default='models', help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    
    args = parser.parse_args()
    
    # Setup config
    config = TrainingConfig()
    config.total_timesteps = args.timesteps
    config.num_envs = args.num_envs
    config.seed = args.seed
    config.model_dir = args.model_dir
    config.log_dir = args.log_dir
    
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device
    
    # Create and run training pipeline
    pipeline = PTIETrainingPipeline(config)
    pipeline.train()


if __name__ == '__main__':
    main()