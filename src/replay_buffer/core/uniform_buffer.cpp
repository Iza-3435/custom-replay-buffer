#include "../../include/core/uniform_buffer.hpp"
#include <algorithm>
#include <mutex>
#include <chrono>

namespace replay_buffer {
namespace core {

template<typename StateType, typename ActionType>
UniformReplayBuffer<StateType, ActionType>::UniformReplayBuffer(
    size_t capacity, const ReplayBufferConfig& config)
    : capacity_(capacity), head_(0), size_(0), total_additions_(0), total_samples_(0),
      thread_safe_(config.thread_safe), rng_(std::random_device{}()), config_(config) {
    
    buffer_.reserve(capacity_);
    buffer_.resize(capacity_);
    
    if (config_.use_memory_pool) {
        memory_pool_ = std::make_unique<memory::MemoryPool<ExperienceType>>(
            capacity_ + config_.batch_allocation_size);
    }
}

template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::add(const ExperienceType& experience) {
    if (thread_safe_) {
        std::unique_lock<std::mutex> lock(mutex_);
        add_impl<std::unique_lock<std::shared_mutex>>(experience);
    } else {
        add_impl<std::nullptr_t>(experience);
    }
}

template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::add(ExperienceType&& experience) {
    if (thread_safe_) {
        std::unique_lock<std::mutex> lock(mutex_);
        add_impl<std::unique_lock<std::shared_mutex>>(std::move(experience));
    } else {
        add_impl<std::nullptr_t>(std::move(experience));
    }
}

template<typename StateType, typename ActionType>
template<typename LockType>
void UniformReplayBuffer<StateType, ActionType>::add_impl(ExperienceType experience) {
    // Set timestamp if not already set
    if (experience.timestamp == 0) {
        experience.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    
    // Add to buffer using circular indexing
    buffer_[head_] = std::move(experience);
    head_ = (head_ + 1) % capacity_;
    
    // Update size atomically
    size_t current_size = size_.load(std::memory_order_acquire);
    if (current_size < capacity_) {
        size_.store(current_size + 1, std::memory_order_release);
    }
    
    total_additions_.fetch_add(1, std::memory_order_relaxed);
}

template<typename StateType, typename ActionType>
typename UniformReplayBuffer<StateType, ActionType>::BatchType
UniformReplayBuffer<StateType, ActionType>::sample(size_t batch_size) {
    BatchType batch;
    size_t current_size = size();
    
    if (current_size == 0 || batch_size == 0) {
        return batch;
    }
    
    batch_size = std::min(batch_size, current_size);
    batch.reserve(batch_size);
    
    std::uniform_int_distribution<size_t> dist(0, current_size - 1);
    
    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx;
        if (thread_safe_) {
            std::lock_guard<std::mutex> lock(mutex_);
            idx = dist(rng_);
        } else {
            idx = dist(rng_);
        }
        
        const auto& exp = buffer_[idx];
        
        batch.states.push_back(exp.state);
        batch.actions.push_back(exp.action);
        batch.rewards.push_back(exp.reward);
        batch.next_states.push_back(exp.next_state);
        batch.dones.push_back(exp.done);
        batch.priorities.push_back(exp.priority);
        batch.timestamps.push_back(exp.timestamp);
        batch.indices.push_back(idx);
    }
    
    total_samples_.fetch_add(batch_size, std::memory_order_relaxed);
    return batch;
}


template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::update_priorities(
    const std::vector<size_t>& indices, const std::vector<float>& priorities) {
    // No-op for uniform buffer, but we could store priorities for future use
    (void)indices;
    (void)priorities;
}

template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::set_priority_function(PriorityFunction func) {
    // No-op for uniform buffer
    (void)func;
}

template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::clear() {
    if (thread_safe_) {
        std::unique_lock<std::mutex> lock(mutex_);
    }
    head_ = 0;
    size_.store(0, std::memory_order_release);
    // Note: We don't clear the actual buffer data for performance reasons
}

template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::reserve(size_t new_capacity) {
    if (new_capacity <= capacity_) return;
    
    if (thread_safe_) {
        std::unique_lock<std::mutex> lock(mutex_);
    }
    
    std::vector<ExperienceType> new_buffer(new_capacity);
    size_t current_size = size();
    
    // Copy existing data maintaining order
    if (current_size > 0) {
        size_t start_idx = (head_ >= current_size) ? head_ - current_size : 
                          capacity_ - (current_size - head_);
        
        for (size_t i = 0; i < current_size; ++i) {
            new_buffer[i] = std::move(buffer_[(start_idx + i) % capacity_]);
        }
    }
    
    buffer_ = std::move(new_buffer);
    capacity_ = new_capacity;
    head_ = current_size % capacity_;
}

template<typename StateType, typename ActionType>
void UniformReplayBuffer<StateType, ActionType>::set_thread_safe(bool enable) {
    thread_safe_ = enable;
}


// Explicit template instantiations for common types
template class UniformReplayBuffer<std::vector<float>, int>;
template class UniformReplayBuffer<std::vector<std::vector<float>>, std::vector<float>>;

} // namespace core
} // namespace replay_buffer