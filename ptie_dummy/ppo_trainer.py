"""
PPO Trainer with GAE for PTIE Dummy Implementation
Implements the training loop following PerfectDou's approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from .networks import PTIEActorCritic
from .feature_encoder import DummyFeatureEncoder, FeatureType
from .reward_calculator import OracleRewardCalculator
from game.game import DummyGame, TurnDirection
from game.player import Player


class PPOConfig:
    """Configuration for PPO training."""
    
    def __init__(self):
        # PPO hyperparameters (following PerfectDou)
        self.learning_rate = 1e-4  # Reduced from 3e-4 to prevent NaN
        self.epsilon_clip = 0.2
        self.entropy_weight = 0.1
        self.value_loss_weight = 0.5
        self.max_grad_norm = 0.5
        
        # GAE parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Training parameters
        self.batch_size = 64
        self.ppo_epochs = 4
        self.num_workers = 8
        self.update_timesteps = 2048
        
        # Network architecture
        self.policy_hidden_layers = [256, 256, 256, 512]
        self.value_hidden_layers = [256, 256, 256, 256]


class TrajectoryBuffer:
    """Buffer for storing trajectory data during rollouts."""
    
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.imperfect_features = []
        self.perfect_features = []
        self.legal_actions = []
        self.actions = []
        self.action_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
    def add_step(self, imperfect_features: Dict[str, np.ndarray],
                 perfect_features: Dict[str, np.ndarray],
                 legal_actions: np.ndarray,
                 action: int,
                 action_log_prob: float,
                 reward: float,
                 value: float,
                 done: bool):
        """Add a single step to the buffer."""
        self.imperfect_features.append(imperfect_features)
        self.perfect_features.append(perfect_features)
        self.legal_actions.append(legal_actions)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def compute_gae(self, config: PPOConfig, last_value: float = 0.0):
        """Compute GAE advantages and returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [True])
        
        deltas = rewards + config.gamma * values[1:] * (1 - dones[1:]) - values[:-1]
        
        advantages = np.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + config.gamma * config.gae_lambda * (1 - dones[t+1]) * advantage
            advantages[t] = advantage
            
        returns = advantages + values[:-1]
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()
        
    def get_batches(self, batch_size: int, device: str = 'cpu'):
        """Generate training batches."""
        indices = np.arange(len(self.actions))
        np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            # Pre-convert to numpy arrays for efficiency
            imperfect_card_features = np.array([self.imperfect_features[i]['card_features'] for i in batch_indices])
            imperfect_game_state = np.array([self.imperfect_features[i]['game_state'] for i in batch_indices])
            perfect_card_features = np.array([self.perfect_features[i]['card_features'] for i in batch_indices])
            perfect_game_state = np.array([self.perfect_features[i]['game_state'] for i in batch_indices])
            legal_actions_array = np.array([self.legal_actions[i] for i in batch_indices])
            actions_array = np.array([self.actions[i] for i in batch_indices])
            old_log_probs_array = np.array([self.action_log_probs[i] for i in batch_indices])
            advantages_array = np.array([self.advantages[i] for i in batch_indices])
            returns_array = np.array([self.returns[i] for i in batch_indices])
            
            batch = {}
            batch['imperfect_features'] = {
                'card_features': torch.from_numpy(imperfect_card_features).float().to(device),
                'game_state': torch.from_numpy(imperfect_game_state).float().to(device)
            }
            batch['perfect_features'] = {
                'card_features': torch.from_numpy(perfect_card_features).float().to(device),
                'game_state': torch.from_numpy(perfect_game_state).float().to(device)
            }
            batch['legal_actions'] = torch.from_numpy(legal_actions_array).float().to(device)
            batch['actions'] = torch.from_numpy(actions_array).long().to(device)
            batch['old_log_probs'] = torch.from_numpy(old_log_probs_array).float().to(device)
            batch['advantages'] = torch.from_numpy(advantages_array).float().to(device)
            batch['returns'] = torch.from_numpy(returns_array).float().to(device)
            
            yield batch


class PPOTrainer:
    """PPO Trainer for PTIE Dummy implementation."""
    
    def __init__(self, config: PPOConfig = None, device: str = 'cpu'):
        self.config = config or PPOConfig()
        self.device = device
        
        # Initialize components
        self.feature_encoder = DummyFeatureEncoder()
        self.reward_calculator = OracleRewardCalculator()
        
        # Get feature dimensions
        imperfect_card_dim, imperfect_state_dim = self.feature_encoder.get_feature_dimensions(FeatureType.IMPERFECT)
        perfect_card_dim, perfect_state_dim = self.feature_encoder.get_feature_dimensions(FeatureType.PERFECT)
        
        # Initialize network
        self.model = PTIEActorCritic(
            imperfect_card_dim, imperfect_state_dim,
            perfect_card_dim, perfect_state_dim
        ).to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Training statistics
        self.step_count = 0
        self.episode_count = 0
        self.stats = defaultdict(list)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def collect_rollouts(self, games: List[DummyGame], num_steps: int) -> TrajectoryBuffer:
        """
        Collect rollouts from multiple game environments.
        
        Args:
            games: List of game instances for parallel rollout
            num_steps: Number of steps to collect
            
        Returns:
            TrajectoryBuffer with collected data
        """
        buffer = TrajectoryBuffer()
        self.model.eval()
        
        active_games = [(game, game.current_player) for game in games if game.game_state.name == 'STARTED']
        
        with torch.no_grad():
            for step in range(num_steps):
                if not active_games:
                    # If no active games, reset and continue
                    for game in games:
                        try:
                            game.reset()
                            game.start_game(TurnDirection.CLOCKWISE, [True, True, True, True], 0)
                            game.deal_cards()
                            game.save_initial_state()
                            game.next_turn()
                        except:
                            pass
                    active_games = [(game, game.current_player) for game in games if game.game_state.name == 'STARTED']
                    if not active_games:
                        break
                    
                # Collect data for current step
                step_data = []
                
                for game, current_player in active_games:
                    # Encode features
                    imperfect_features = self.feature_encoder.encode_game_state(
                        game, current_player, FeatureType.IMPERFECT
                    )
                    perfect_features = self.feature_encoder.encode_game_state(
                        game, current_player, FeatureType.PERFECT
                    )
                    legal_actions, legal_actions_list = self.feature_encoder.encode_legal_actions(game, current_player)
                    
                    # Convert to tensors
                    imperfect_tensors = {
                        'card_features': torch.FloatTensor(imperfect_features['card_features']).unsqueeze(0).to(self.device),
                        'game_state': torch.FloatTensor(imperfect_features['game_state']).unsqueeze(0).to(self.device)
                    }
                    perfect_tensors = {
                        'card_features': torch.FloatTensor(perfect_features['card_features']).unsqueeze(0).to(self.device),
                        'game_state': torch.FloatTensor(perfect_features['game_state']).unsqueeze(0).to(self.device)
                    }
                    legal_actions_tensor = torch.FloatTensor(legal_actions).unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    outputs = self.model.forward(
                        imperfect_tensors, perfect_tensors, legal_actions_tensor
                    )
                    
                    # Sample action
                    action_dist = torch.distributions.Categorical(
                        torch.softmax(outputs['action_logits'], dim=-1)
                    )
                    action = action_dist.sample()
                    action_log_prob = action_dist.log_prob(action)
                    
                    # Get value estimate
                    value = outputs['state_value'].squeeze()
                    
                    step_data.append({
                        'game': game,
                        'player': current_player,
                        'imperfect_features': imperfect_features,
                        'perfect_features': perfect_features,
                        'legal_actions': legal_actions,
                        'legal_actions_list': legal_actions_list,
                        'action': action.item(),
                        'action_log_prob': action_log_prob.item(),
                        'value': value.item()
                    })
                
                # Execute actions and collect rewards
                new_active_games = []
                
                for data in step_data:
                    game = data['game']
                    player = data['player']
                    action_idx = data['action']
                    
                    # Get actual action from legal actions
                    legal_actions_list = data['legal_actions_list']
                    if action_idx < len(legal_actions_list):
                        action = legal_actions_list[action_idx]
                        
                        # Calculate reward before action
                        old_state_value = self.reward_calculator.calculate_oracle_reward(game, player)
                        
                        # Execute action
                        try:
                            game.take_action(action)
                            
                            # Calculate reward after action
                            new_state_value = self.reward_calculator.calculate_oracle_reward(game, player)
                            reward = new_state_value - old_state_value
                            
                            # Check if game is done
                            done = game.game_state.name == 'FINISHED'
                            
                        except Exception as e:
                            self.logger.warning(f"Action execution failed: {e}")
                            reward = -1.0  # Penalty for invalid action
                            done = True
                    else:
                        reward = -1.0  # Penalty for invalid action selection
                        done = True
                    
                    # Add to buffer
                    buffer.add_step(
                        data['imperfect_features'],
                        data['perfect_features'],
                        data['legal_actions'],
                        data['action'],
                        data['action_log_prob'],
                        reward,
                        data['value'],
                        done
                    )
                    
                    # Update active games
                    if not done and game.game_state.name == 'STARTED':
                        new_active_games.append((game, game.current_player))
                
                active_games = new_active_games
                self.step_count += 1
        
        # Compute advantages and returns
        buffer.compute_gae(self.config)
        
        return buffer
    
    def update_policy(self, buffer: TrajectoryBuffer) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Args:
            buffer: Trajectory buffer with rollout data
            
        Returns:
            Training statistics
        """
        self.model.train()
        
        total_policy_loss = 0
        total_value_loss = 0  
        total_entropy_loss = 0
        
        for epoch in range(self.config.ppo_epochs):
            for batch in buffer.get_batches(self.config.batch_size, self.device):
                # Forward pass
                outputs = self.model.forward(
                    batch['imperfect_features'],
                    batch['perfect_features'],
                    batch['legal_actions']
                )
                
                # Calculate policy loss
                action_dist = torch.distributions.Categorical(
                    torch.softmax(outputs['action_logits'], dim=-1)
                )
                new_log_probs = action_dist.log_prob(batch['actions'])
                entropy = action_dist.entropy().mean()
                
                # PPO clipping
                ratio = torch.exp(new_log_probs - batch['old_log_probs'])
                advantages = batch['advantages']
                
                # Normalize advantages with safeguards
                adv_mean = advantages.mean()
                adv_std = advantages.std()
                if adv_std > 1e-6:  # Only normalize if std is significant
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                else:
                    advantages = advantages - adv_mean  # Just center, don't normalize
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon_clip, 1 + self.config.epsilon_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss  
                value_pred = outputs['state_value'].squeeze(-1)  # Remove last dimension only
                returns = batch['returns']
                
                # Ensure same shape
                if value_pred.shape != returns.shape:
                    if value_pred.dim() == 0:  # Scalar
                        value_pred = value_pred.unsqueeze(0)
                    if returns.dim() == 0:  # Scalar  
                        returns = returns.unsqueeze(0)
                
                value_loss = nn.MSELoss()(value_pred, returns)
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.config.value_loss_weight * value_loss - 
                    self.config.entropy_weight * entropy
                )
                
                # Check for NaN in losses before backward pass
                if torch.isnan(total_loss).any():
                    self.logger.warning(f"NaN detected in total_loss: {total_loss}, skipping batch")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        self.logger.warning(f"NaN gradient in {name}")
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    self.logger.warning("Skipping optimizer step due to NaN gradients")
                    continue
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.item()
        
        # Calculate average losses
        num_updates = self.config.ppo_epochs * len(list(buffer.get_batches(self.config.batch_size, self.device)))
        
        # Prevent division by zero
        if num_updates == 0:
            num_updates = 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy_loss / num_updates,
        }
    
    def train_step(self, games: List[DummyGame]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            games: List of game environments
            
        Returns:
            Training statistics
        """
        # Collect rollouts
        buffer = self.collect_rollouts(games, self.config.update_timesteps)
        
        # Update policy
        train_stats = self.update_policy(buffer)
        
        # Update statistics
        self.stats['policy_loss'].append(train_stats['policy_loss'])
        self.stats['value_loss'].append(train_stats['value_loss'])
        self.stats['entropy'].append(train_stats['entropy'])
        
        # Log progress
        if self.step_count % 1000 == 0:
            self.logger.info(f"Step {self.step_count}: "
                           f"Policy Loss: {train_stats['policy_loss']:.4f}, "
                           f"Value Loss: {train_stats['value_loss']:.4f}, "
                           f"Entropy: {train_stats['entropy']:.4f}")
        
        return train_stats
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'stats': dict(self.stats)
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.stats = defaultdict(list, checkpoint['stats'])