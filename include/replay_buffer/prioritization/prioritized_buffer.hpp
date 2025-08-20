#pragma once

#include "../core/replay_buffer_interface.hpp"
#include "sum_tree.hpp"
#include "../concurrency/spinlock.hpp"
#include <shared_mutex>
#include <functional>

namespace replay_buffer {
namespace prioritization {

template<typename StateType, typename ActionType>
class PrioritizedReplayBuffer : public core::IReplayBuffer<StateType, ActionType> {
public:
    using ExperienceType = typename core::IReplayBuffer<StateType, ActionType>::ExperienceType;
    using BatchType = typename core::IReplayBuffer<StateType, ActionType>::BatchType;
    using PriorityFunction = typename core::IReplayBuffer<StateType, ActionType>::PriorityFunction;

private:
    std::vector<ExperienceType> buffer_;
    std::unique_ptr<SumTree> sum_tree_;
    size_t capacity_;
    size_t head_;
    std::atomic<size_t> size_;
    std::atomic<size_t> total_additions_;
    std::atomic<size_t> total_samples_;
    
    // Priority parameters
    double alpha_;      // Prioritization exponent
    double beta_;       // Importance sampling exponent
    double beta_increment_; // Beta annealing rate
    double epsilon_;    // Small constant to prevent zero priorities
    double max_priority_; // Track maximum priority for new experiences
    
    // Thread safety
    mutable std::shared_mutex mutex_;
    bool thread_safe_;
    
    // Random number generation
    mutable std::mt19937 rng_;
    
    // Priority function
    PriorityFunction priority_func_;
    
    // Configuration
    core::ReplayBufferConfig config_;

public:
    explicit PrioritizedReplayBuffer(size_t capacity, const core::ReplayBufferConfig& config = {});
    ~PrioritizedReplayBuffer() override = default;

    // Core operations
    void add(const ExperienceType& experience) override;
    void add(ExperienceType&& experience) override;
    BatchType sample(size_t batch_size) override;
    
    // Buffer state
    size_t size() const override { return size_.load(std::memory_order_acquire); }
    size_t capacity() const override { return capacity_; }
    bool empty() const override { return size() == 0; }
    bool full() const override { return size() == capacity_; }
    
    // Priority management
    void update_priorities(const std::vector<size_t>& indices, 
                          const std::vector<float>& priorities) override;
    void set_priority_function(PriorityFunction func) override;
    
    // Buffer management
    void clear() override;
    void reserve(size_t capacity) override;
    
    // Statistics
    double get_avg_priority() const override;
    size_t get_total_additions() const override { return total_additions_.load(); }
    size_t get_total_samples() const override { return total_samples_.load(); }
    
    // Thread safety control
    void set_thread_safe(bool enable) override;
    bool is_thread_safe() const override { return thread_safe_; }
    
    // Prioritized-specific methods
    void set_alpha(double alpha) { alpha_ = alpha; }
    void set_beta(double beta) { beta_ = beta; }
    void set_beta_increment(double increment) { beta_increment_ = increment; }
    void anneal_beta() { beta_ = std::min(1.0, beta_ + beta_increment_); }
    
    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    double get_max_priority() const { return max_priority_; }
    double get_total_priority() const;

private:
    template<typename LockType>
    void add_impl(ExperienceType experience);
    
    void clear_impl();
    void reserve_impl(size_t new_capacity);
    
    double calculate_priority(const ExperienceType& experience) const;
    double default_priority_function(const ExperienceType& experience) const;
    
    size_t get_buffer_index(size_t tree_index) const;
    size_t get_tree_index(size_t buffer_index) const;
};

template<typename StateType, typename ActionType>
PrioritizedReplayBuffer<StateType, ActionType>::PrioritizedReplayBuffer(
    size_t capacity, const core::ReplayBufferConfig& config)
    : capacity_(capacity), head_(0), size_(0), total_additions_(0), total_samples_(0),
      alpha_(config.alpha), beta_(config.beta), beta_increment_(config.beta_increment),
      epsilon_(config.priority_epsilon), max_priority_(1.0),
      thread_safe_(config.thread_safe), rng_(std::random_device{}()),
      config_(config) {
    
    buffer_.resize(capacity_);
    sum_tree_ = std::make_unique<SumTree>(capacity_);
    
    // Set default priority function
    priority_func_ = [this](const ExperienceType& exp) {
        return default_priority_function(exp);
    };
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::add(const ExperienceType& experience) {
    if (thread_safe_) {
        std::unique_lock lock(mutex_);
        add_impl<std::unique_lock<std::shared_mutex>>(experience);
    } else {
        add_impl<std::nullptr_t>(experience);
    }
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::add(ExperienceType&& experience) {
    if (thread_safe_) {
        std::unique_lock lock(mutex_);
        add_impl<std::unique_lock<std::shared_mutex>>(std::move(experience));
    } else {
        add_impl<std::nullptr_t>(std::move(experience));
    }
}

template<typename StateType, typename ActionType>
template<typename LockType>
void PrioritizedReplayBuffer<StateType, ActionType>::add_impl(ExperienceType experience) {
    // Set timestamp if not already set
    if (experience.timestamp == 0) {
        experience.timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }
    
    // Calculate priority for new experience
    double priority = calculate_priority(experience);
    max_priority_ = std::max(max_priority_, priority);
    
    // Add to buffer using circular indexing
    buffer_[head_] = std::move(experience);
    
    // Set priority in sum tree
    sum_tree_->set(head_, std::pow(priority + epsilon_, alpha_));
    
    head_ = (head_ + 1) % capacity_;
    
    // Update size atomically
    size_t current_size = size_.load(std::memory_order_acquire);
    if (current_size < capacity_) {
        size_.store(current_size + 1, std::memory_order_release);
    }
    
    total_additions_.fetch_add(1, std::memory_order_relaxed);
}

template<typename StateType, typename ActionType>
typename PrioritizedReplayBuffer<StateType, ActionType>::BatchType
PrioritizedReplayBuffer<StateType, ActionType>::sample(size_t batch_size) {
    BatchType batch;
    size_t current_size = size();
    
    if (current_size == 0 || batch_size == 0) {
        return batch;
    }
    
    batch_size = std::min(batch_size, current_size);
    batch.reserve(batch_size);
    
    if (thread_safe_) {
        std::shared_lock lock(mutex_);
        sample_impl(batch, batch_size, current_size);
    } else {
        sample_impl(batch, batch_size, current_size);
    }
    
    total_samples_.fetch_add(batch_size, std::memory_order_relaxed);
    return batch;
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::sample_impl(
    BatchType& batch, size_t batch_size, size_t current_size) {
    
    double total_priority = sum_tree_->total();
    double min_priority = sum_tree_->min_priority();
    
    if (total_priority <= 0.0) {
        // Fallback to uniform sampling
        std::uniform_int_distribution<size_t> dist(0, current_size - 1);
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = dist(rng_);
            const auto& exp = buffer_[idx];
            
            batch.states.push_back(exp.state);
            batch.actions.push_back(exp.action);
            batch.rewards.push_back(exp.reward);
            batch.next_states.push_back(exp.next_state);
            batch.dones.push_back(exp.done);
            batch.priorities.push_back(1.0f); // Uniform weight
            batch.timestamps.push_back(exp.timestamp);
            batch.indices.push_back(idx);
        }
        return;
    }
    
    // Calculate maximum importance sampling weight
    double max_weight = std::pow((min_priority + epsilon_) / total_priority, -beta_);
    
    std::uniform_real_distribution<double> dist(0.0, total_priority);
    
    for (size_t i = 0; i < batch_size; ++i) {
        // Sample from priority distribution
        double sample_value = dist(rng_);
        size_t tree_idx = sum_tree_->sample(sample_value);
        
        // Ensure valid buffer index
        size_t buffer_idx = tree_idx;
        if (buffer_idx >= current_size) {
            buffer_idx = tree_idx % current_size;
        }
        
        const auto& exp = buffer_[buffer_idx];
        
        // Calculate importance sampling weight
        double priority = sum_tree_->get(tree_idx);
        double prob = priority / total_priority;
        double weight = std::pow(prob, -beta_) / max_weight;
        
        batch.states.push_back(exp.state);
        batch.actions.push_back(exp.action);
        batch.rewards.push_back(exp.reward);
        batch.next_states.push_back(exp.next_state);
        batch.dones.push_back(exp.done);
        batch.priorities.push_back(static_cast<float>(weight));
        batch.timestamps.push_back(exp.timestamp);
        batch.indices.push_back(buffer_idx);
    }
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::update_priorities(
    const std::vector<size_t>& indices, const std::vector<float>& priorities) {
    
    if (indices.size() != priorities.size()) return;
    
    if (thread_safe_) {
        std::unique_lock lock(mutex_);
    }
    
    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        double priority = static_cast<double>(priorities[i]);
        
        if (idx < size()) {
            max_priority_ = std::max(max_priority_, priority);
            sum_tree_->set(idx, std::pow(priority + epsilon_, alpha_));
        }
    }
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::set_priority_function(PriorityFunction func) {
    if (thread_safe_) {
        std::unique_lock lock(mutex_);
    }
    priority_func_ = func ? func : [this](const ExperienceType& exp) {
        return default_priority_function(exp);
    };
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::clear() {
    if (thread_safe_) {
        std::unique_lock lock(mutex_);
        clear_impl();
    } else {
        clear_impl();
    }
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::clear_impl() {
    head_ = 0;
    size_.store(0, std::memory_order_release);
    max_priority_ = 1.0;
    
    // Clear sum tree
    for (size_t i = 0; i < capacity_; ++i) {
        sum_tree_->set(i, 0.0);
    }
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::reserve(size_t new_capacity) {
    if (new_capacity <= capacity_) return;
    
    if (thread_safe_) {
        std::unique_lock lock(mutex_);
        reserve_impl(new_capacity);
    } else {
        reserve_impl(new_capacity);
    }
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::reserve_impl(size_t new_capacity) {
    // Create new buffer and sum tree
    std::vector<ExperienceType> new_buffer(new_capacity);
    auto new_sum_tree = std::make_unique<SumTree>(new_capacity);
    
    size_t current_size = size();
    
    // Copy existing data maintaining order
    if (current_size > 0) {
        size_t start_idx = (head_ >= current_size) ? head_ - current_size : 
                          capacity_ - (current_size - head_);
        
        for (size_t i = 0; i < current_size; ++i) {
            size_t old_idx = (start_idx + i) % capacity_;
            new_buffer[i] = std::move(buffer_[old_idx]);
            
            // Copy priority
            double priority = sum_tree_->get(old_idx);
            new_sum_tree->set(i, priority);
        }
    }
    
    buffer_ = std::move(new_buffer);
    sum_tree_ = std::move(new_sum_tree);
    capacity_ = new_capacity;
    head_ = current_size % capacity_;
}

template<typename StateType, typename ActionType>
double PrioritizedReplayBuffer<StateType, ActionType>::get_avg_priority() const {
    double total = get_total_priority();
    size_t current_size = size();
    return current_size > 0 ? total / current_size : 0.0;
}

template<typename StateType, typename ActionType>
double PrioritizedReplayBuffer<StateType, ActionType>::get_total_priority() const {
    if (thread_safe_) {
        std::shared_lock lock(mutex_);
    }
    return sum_tree_->total();
}

template<typename StateType, typename ActionType>
void PrioritizedReplayBuffer<StateType, ActionType>::set_thread_safe(bool enable) {
    thread_safe_ = enable;
}

template<typename StateType, typename ActionType>
double PrioritizedReplayBuffer<StateType, ActionType>::calculate_priority(
    const ExperienceType& experience) const {
    
    if (priority_func_) {
        return std::max(epsilon_, priority_func_(experience));
    }
    return max_priority_; // Use max priority for new experiences
}

template<typename StateType, typename ActionType>
double PrioritizedReplayBuffer<StateType, ActionType>::default_priority_function(
    const ExperienceType& experience) const {
    
    // Simple heuristic: higher reward magnitude = higher priority
    return std::abs(experience.reward) + epsilon_;
}

// Explicit template instantiations for common types
extern template class PrioritizedReplayBuffer<std::vector<float>, int>;
extern template class PrioritizedReplayBuffer<std::vector<std::vector<float>>, std::vector<float>>;

} // namespace prioritization
} // namespace replay_buffer