"""
Neural Network Architectures for PTIE Dummy Implementation
Implements Actor-Critic networks following PerfectDou's design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

from .feature_encoder import FeatureType


class AttentionModule(nn.Module):
    """
    Attention mechanism for action selection.
    Follows PerfectDou's target attention design.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.query_net = nn.Linear(state_dim, hidden_dim)
        self.key_net = nn.Linear(action_dim, hidden_dim)
        self.value_net = nn.Linear(action_dim, hidden_dim)
        
        # Output projection
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state_repr: torch.Tensor, action_features: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of attention mechanism.
        
        Args:
            state_repr: Game state representation [batch_size, state_dim]
            action_features: Legal action features [batch_size, num_actions, action_dim]
            action_mask: Mask for valid actions [batch_size, num_actions]
            
        Returns:
            Action probabilities [batch_size, num_actions]
        """
        batch_size, num_actions, _ = action_features.shape
        
        # Compute attention
        query = self.query_net(state_repr)  # [batch_size, hidden_dim]
        keys = self.key_net(action_features)  # [batch_size, num_actions, hidden_dim]
        values = self.value_net(action_features)  # [batch_size, num_actions, hidden_dim]
        
        # Expand query for attention computation
        query_expanded = query.unsqueeze(1).expand(-1, num_actions, -1)  # [batch_size, num_actions, hidden_dim]
        
        # Attention scores
        attention_scores = torch.sum(query_expanded * keys, dim=2)  # [batch_size, num_actions]
        
        # Clamp attention scores to prevent extreme values
        attention_scores = torch.clamp(attention_scores, min=-10.0, max=10.0)
        
        # Apply mask to attention scores if provided
        if action_mask is not None:
            attention_scores = attention_scores.masked_fill(~action_mask, -1e9)  # Use finite large negative number
        
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, num_actions]
        
        # Apply attention to values
        attended_values = attention_weights.unsqueeze(2) * values  # [batch_size, num_actions, hidden_dim]
        
        # Generate action logits
        action_logits = self.output_net(attended_values).squeeze(2)  # [batch_size, num_actions]
        
        # Apply mask to final logits as well
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, -1e9)  # Use finite large negative number
        
        # Clamp final logits to prevent extreme values
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)
        
        return action_logits


class PolicyNetwork(nn.Module):
    """
    Policy Network (Actor) for PTIE.
    Uses imperfect information only.
    """
    
    def __init__(self, card_feature_dim: int, game_state_dim: int, action_dim: int = 100):
        super().__init__()
        
        self.card_feature_dim = card_feature_dim
        self.game_state_dim = game_state_dim
        self.action_dim = action_dim
        
        # Feature encoding layers
        self.card_encoder = nn.Sequential(
            nn.Linear(card_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # LSTM for sequential processing (following PerfectDou)
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
        # Game state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Combined feature processor
        self.feature_combiner = nn.Sequential(
            nn.Linear(256 + 256, 512),  # LSTM output + state features
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Attention mechanism for action selection
        self.attention = AttentionModule(256, action_dim, 256)
        
    def forward(self, card_features: torch.Tensor, game_state: torch.Tensor, 
                legal_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of policy network.
        
        Args:
            card_features: Card feature matrix [batch_size, card_feature_dim]
            game_state: Game state features [batch_size, game_state_dim]
            legal_actions: Legal action features [batch_size, num_actions, action_dim]
            
        Returns:
            action_logits: Action probability logits [batch_size, num_actions]
            state_repr: Internal state representation for value estimation
        """
        batch_size = card_features.shape[0]
        
        # Encode card features
        card_repr = self.card_encoder(card_features)  # [batch_size, 256]
        
        # Process through LSTM (treating as sequence of length 1 for now)
        card_repr = card_repr.unsqueeze(1)  # [batch_size, 1, 256]
        lstm_out, _ = self.lstm(card_repr)
        lstm_repr = lstm_out.squeeze(1)  # [batch_size, 256]
        
        # Encode game state
        state_repr = self.state_encoder(game_state)  # [batch_size, 256]
        
        # Combine features
        combined_repr = torch.cat([lstm_repr, state_repr], dim=1)  # [batch_size, 512]
        state_representation = self.feature_combiner(combined_repr)  # [batch_size, 256]
        
        # Create action mask (valid actions have non-zero features)
        action_mask = (legal_actions.sum(dim=2) != 0)  # [batch_size, num_actions]
        
        # Apply attention for action selection
        action_logits = self.attention(state_representation, legal_actions, action_mask)  # [batch_size, num_actions]
        
        return action_logits, state_representation


class ValueNetwork(nn.Module):
    """
    Value Network (Critic) for PTIE.
    Uses perfect information for accurate state evaluation.
    """
    
    def __init__(self, card_feature_dim: int, game_state_dim: int):
        super().__init__()
        
        self.card_feature_dim = card_feature_dim
        self.game_state_dim = game_state_dim
        
        # Imperfect information encoder (for 780-dim imperfect features)
        self.imperfect_card_dim = 15 * 13 * 4  # 780 dimensions
        self.imperfect_encoder = nn.Sequential(
            nn.Linear(self.imperfect_card_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Perfect information encoder (separate pathway)
        self.perfect_encoder = nn.Sequential(
            nn.Linear(card_feature_dim, 512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Game state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(game_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Value estimation layers (following PerfectDou's MLP design)
        self.value_head = nn.Sequential(
            nn.Linear(256 + 256 + 256, 256),  # imperfect + perfect + game_state
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Single value output
        )
        
    def forward(self, perfect_card_features: torch.Tensor, perfect_game_state: torch.Tensor,
                imperfect_card_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of value network.
        
        Args:
            perfect_card_features: Perfect information card features [batch_size, card_feature_dim]
            perfect_game_state: Perfect information game state [batch_size, game_state_dim]
            imperfect_card_features: Imperfect card features for shared encoding
            
        Returns:
            state_value: Estimated state value [batch_size, 1]
        """
        batch_size = perfect_card_features.shape[0]
        
        # Extract imperfect features from perfect features if not provided separately
        if imperfect_card_features is None:
            # Perfect features contain imperfect features in first 15*13*4=780 dimensions
            # Perfect features: 17*13*4=884, Imperfect features: 15*13*4=780
            imperfect_card_features = perfect_card_features[:, :self.imperfect_card_dim]
        
        # Encode imperfect information
        imperfect_repr = self.imperfect_encoder(imperfect_card_features)  # [batch_size, 256]
        
        # Encode perfect information
        perfect_repr = self.perfect_encoder(perfect_card_features)  # [batch_size, 256]
        
        # Encode game state
        state_repr = self.state_encoder(perfect_game_state)  # [batch_size, 256]
        
        # Combine all representations
        combined_repr = torch.cat([imperfect_repr, perfect_repr, state_repr], dim=1)  # [batch_size, 768]
        
        # Estimate value
        state_value = self.value_head(combined_repr)  # [batch_size, 1]
        
        return state_value


class PTIEActorCritic(nn.Module):
    """
    Combined Actor-Critic Network for PTIE.
    Implements the Perfect-Training-Imperfect-Execution paradigm.
    """
    
    def __init__(self, imperfect_card_dim: int, imperfect_state_dim: int,
                 perfect_card_dim: int, perfect_state_dim: int, action_dim: int = 100):
        super().__init__()
        
        # Policy network (uses imperfect information)
        self.policy_net = PolicyNetwork(imperfect_card_dim, imperfect_state_dim, action_dim)
        
        # Value network (uses perfect information)
        self.value_net = ValueNetwork(perfect_card_dim, perfect_state_dim)
        
        # Store dimensions for reference
        self.imperfect_card_dim = imperfect_card_dim
        self.imperfect_state_dim = imperfect_state_dim
        self.perfect_card_dim = perfect_card_dim
        self.perfect_state_dim = perfect_state_dim
        self.action_dim = action_dim
        
    def forward(self, imperfect_features: Dict[str, torch.Tensor], 
                perfect_features: Dict[str, torch.Tensor],
                legal_actions: torch.Tensor,
                mode: str = "both") -> Dict[str, torch.Tensor]:
        """
        Forward pass through both networks.
        
        Args:
            imperfect_features: Dictionary with 'card_features' and 'game_state'
            perfect_features: Dictionary with 'card_features' and 'game_state'  
            legal_actions: Legal action features [batch_size, num_actions, action_dim]
            mode: "policy", "value", or "both"
            
        Returns:
            Dictionary with network outputs
        """
        outputs = {}
        
        if mode in ["policy", "both"]:
            # Policy network forward pass
            action_logits, state_repr = self.policy_net(
                imperfect_features['card_features'],
                imperfect_features['game_state'],
                legal_actions
            )
            outputs['action_logits'] = action_logits
            outputs['state_representation'] = state_repr
            
        if mode in ["value", "both"]:
            # Value network forward pass
            # Pass imperfect features separately if available for better encoding
            imperfect_card_features = None
            if imperfect_features is not None:
                imperfect_card_features = imperfect_features['card_features']
            
            state_value = self.value_net(
                perfect_features['card_features'],
                perfect_features['game_state'],
                imperfect_card_features
            )
            outputs['state_value'] = state_value
            
        return outputs
    
    def get_action_distribution(self, imperfect_features: Dict[str, torch.Tensor],
                               legal_actions: torch.Tensor) -> torch.distributions.Categorical:
        """
        Get action distribution for sampling during execution (imperfect information only).
        
        Args:
            imperfect_features: Imperfect information features
            legal_actions: Legal action features
            
        Returns:
            Categorical distribution over actions
        """
        outputs = self.forward(
            imperfect_features, 
            None,  # No perfect features needed for policy
            legal_actions, 
            mode="policy"
        )
        
        # Add numerical stability and NaN checking
        action_logits = outputs['action_logits']
        
        # Check for NaN in logits
        if torch.isnan(action_logits).any():
            logging.warning("NaN detected in action_logits, using uniform distribution")
            # Fallback to uniform distribution over valid actions
            action_probs = torch.ones_like(action_logits)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        else:
            # Numerical stable softmax with clamping
            action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Check for NaN in probabilities
            if torch.isnan(action_probs).any():
                logging.warning("NaN detected in action_probs, using uniform distribution")
                action_probs = torch.ones_like(action_logits)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        return torch.distributions.Categorical(action_probs)
    
    def evaluate_state(self, perfect_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluate state value using perfect information.
        
        Args:
            perfect_features: Perfect information features
            
        Returns:
            State value estimate
        """
        outputs = self.forward(
            None,  # No imperfect features needed for value
            perfect_features,
            None,  # No actions needed for value
            mode="value"
        )
        
        return outputs['state_value']