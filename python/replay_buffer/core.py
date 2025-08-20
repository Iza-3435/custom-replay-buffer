"""
Core replay buffer interface and implementations.
"""

import numpy as np
from typing import List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum

class BufferType(Enum):
    """Available buffer implementations."""
    SIMPLE = "simple"        # Basic FIFO buffer
    UNIFORM = "uniform"      # Uniform random sampling  
    PRIORITIZED = "prioritized"  # Priority-based sampling
    LOCKFREE = "lockfree"    # Ultra-fast lock-free buffer (recommended)

@dataclass
class Experience:
    """Single experience tuple for reinforcement learning."""
    state: np.ndarray
    action: Union[int, np.ndarray]
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0
    timestamp_ns: int = 0

@dataclass  
class ExperienceBatch:
    """Batch of experiences for training."""
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    priorities: np.ndarray
    indices: np.ndarray

class ReplayBuffer:
    """
    High-performance replay buffer with multiple implementation options.
    
    This is the main interface for the replay buffer library. It automatically
    selects the optimal C++ backend based on the buffer type specified.
    
    Args:
        capacity: Maximum number of experiences to store
        buffer_type: Type of buffer implementation to use
        alpha: Priority exponent (for prioritized buffer)
        beta: Importance sampling exponent (for prioritized buffer)
        seed: Random seed for reproducibility
        
    Example:
        >>> buffer = ReplayBuffer(capacity=100000, buffer_type='lockfree')
        >>> buffer.add(state, action, reward, next_state, done)
        >>> batch = buffer.sample(32)
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 buffer_type: Union[str, BufferType] = BufferType.LOCKFREE,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 seed: Optional[int] = None):
        
        if isinstance(buffer_type, str):
            buffer_type = BufferType(buffer_type.lower())
            
        self.capacity = capacity
        self.buffer_type = buffer_type
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        
        # Initialize backend buffer
        self._init_backend()
        
        # Performance tracking
        self.total_adds = 0
        self.total_samples = 0
        
    def _init_backend(self):
        """Initialize the appropriate C++ backend."""
        try:
            from .bindings import HFTReplayBufferCpp
            self._backend = HFTReplayBufferCpp(
                capacity=self.capacity,
                alpha=self.alpha,
                beta=self.beta
            )
            self._use_cpp = True
        except ImportError:
            # Fallback to Python implementation
            self._init_python_backend()
            self._use_cpp = False
            
    def _init_python_backend(self):
        """Fallback Python implementation."""
        self._buffer = []
        self._position = 0
        self._use_cpp = False
        
    def add(self, 
            state: np.ndarray, 
            action: Union[int, np.ndarray], 
            reward: float,
            next_state: np.ndarray, 
            done: bool,
            priority: float = 1.0) -> None:
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state observed
            done: Whether episode ended
            priority: Priority for sampling (prioritized buffer only)
        """
        if self._use_cpp:
            self._backend.add(state, action, reward, next_state, done, 
                            priority=priority)
        else:
            # Python fallback
            experience = Experience(state, action, reward, next_state, done, priority)
            
            if len(self._buffer) < self.capacity:
                self._buffer.append(experience)
            else:
                self._buffer[self._position] = experience
                self._position = (self._position + 1) % self.capacity
                
        self.total_adds += 1
        
    def sample(self, batch_size: int) -> ExperienceBatch:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            ExperienceBatch containing sampled experiences
        """
        if self._use_cpp:
            return self._backend.sample(batch_size)
        else:
            # Python fallback
            if len(self._buffer) < batch_size:
                indices = list(range(len(self._buffer)))
            else:
                indices = np.random.choice(len(self._buffer), batch_size, replace=False)
                
            experiences = [self._buffer[i] for i in indices]
            
            # Convert to batch format
            states = np.array([exp.state for exp in experiences])
            actions = np.array([exp.action for exp in experiences])
            rewards = np.array([exp.reward for exp in experiences])
            next_states = np.array([exp.next_state for exp in experiences])
            dones = np.array([exp.done for exp in experiences])
            priorities = np.array([exp.priority for exp in experiences])
            
            self.total_samples += 1
            
            return ExperienceBatch(
                states=states,
                actions=actions, 
                rewards=rewards,
                next_states=next_states,
                dones=dones,
                priorities=priorities,
                indices=np.array(indices)
            )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        if self._use_cpp:
            return len(self._backend)
        else:
            return len(self._buffer)
            
    def clear(self) -> None:
        """Clear all experiences from buffer."""
        if self._use_cpp:
            self._backend.clear()
        else:
            self._buffer.clear()
            self._position = 0
            
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        if self._use_cpp:
            return self._backend.get_performance_stats()
        else:
            return {
                'total_adds': self.total_adds,
                'total_samples': self.total_samples,
                'buffer_size': len(self._buffer),
                'capacity': self.capacity,
                'backend': 'python_fallback'
            }