#pragma once

#include "experience.hpp"
#include <vector>
#include <random>
#include <mutex>
#include <atomic>

namespace replay_buffer {
namespace core {

// Simplified buffer for quick testing
template<typename StateType, typename ActionType>
class SimpleReplayBuffer {
public:
    using ExperienceType = Experience<StateType, ActionType>;
    using BatchType = ExperienceBatch<StateType, ActionType>;

private:
    std::vector<ExperienceType> buffer_;
    size_t capacity_;
    size_t head_;
    std::atomic<size_t> size_;
    mutable std::mutex mutex_;
    mutable std::mt19937 rng_;

public:
    explicit SimpleReplayBuffer(size_t capacity) 
        : capacity_(capacity), head_(0), size_(0), rng_(std::random_device{}()) {
        buffer_.resize(capacity_);
    }

    void add(const ExperienceType& experience) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        buffer_[head_] = experience;
        head_ = (head_ + 1) % capacity_;
        
        size_t current_size = size_.load();
        if (current_size < capacity_) {
            size_.store(current_size + 1);
        }
    }

    void add(ExperienceType&& experience) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        buffer_[head_] = std::move(experience);
        head_ = (head_ + 1) % capacity_;
        
        size_t current_size = size_.load();
        if (current_size < capacity_) {
            size_.store(current_size + 1);
        }
    }

    BatchType sample(size_t batch_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        BatchType batch;
        size_t current_size = size_.load();
        
        if (current_size == 0 || batch_size == 0) {
            return batch;
        }
        
        batch_size = std::min(batch_size, current_size);
        batch.reserve(batch_size);
        
        std::uniform_int_distribution<size_t> dist(0, current_size - 1);
        
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = dist(rng_);
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
        
        return batch;
    }

    size_t size() const { return size_.load(); }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size() == 0; }
    bool full() const { return size() == capacity_; }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        head_ = 0;
        size_.store(0);
    }
};

} // namespace core
} // namespace replay_buffer