#pragma once

#include "replay_buffer_interface.hpp"
#include "../memory/memory_pool.hpp"
#include "../concurrency/thread_safe_queue.hpp"
#include <random>
#include <vector>
#include <atomic>
#include <mutex>

namespace replay_buffer {
namespace core {

template<typename StateType, typename ActionType>
class UniformReplayBuffer : public IReplayBuffer<StateType, ActionType> {
public:
    using ExperienceType = typename IReplayBuffer<StateType, ActionType>::ExperienceType;
    using BatchType = typename IReplayBuffer<StateType, ActionType>::BatchType;
    using PriorityFunction = typename IReplayBuffer<StateType, ActionType>::PriorityFunction;

private:
    std::vector<ExperienceType> buffer_;
    size_t capacity_;
    size_t head_;
    std::atomic<size_t> size_;
    std::atomic<size_t> total_additions_;
    std::atomic<size_t> total_samples_;
    
    // Thread safety
    mutable std::mutex mutex_;
    bool thread_safe_;
    
    // Random number generation
    mutable std::mt19937 rng_;
    
    // Memory optimization
    std::unique_ptr<memory::MemoryPool<ExperienceType>> memory_pool_;
    ReplayBufferConfig config_;

public:
    explicit UniformReplayBuffer(size_t capacity, const ReplayBufferConfig& config = {});
    ~UniformReplayBuffer() override = default;

    // Core operations
    void add(const ExperienceType& experience) override;
    void add(ExperienceType&& experience) override;
    BatchType sample(size_t batch_size) override;
    
    // Buffer state
    size_t size() const override { return size_.load(std::memory_order_acquire); }
    size_t capacity() const override { return capacity_; }
    bool empty() const override { return size() == 0; }
    bool full() const override { return size() == capacity_; }
    
    // Priority management (no-op for uniform buffer)
    void update_priorities(const std::vector<size_t>& indices, 
                          const std::vector<float>& priorities) override;
    void set_priority_function(PriorityFunction func) override;
    
    // Buffer management
    void clear() override;
    void reserve(size_t capacity) override;
    
    // Statistics
    double get_avg_priority() const override { return 1.0; }
    size_t get_total_additions() const override { return total_additions_.load(); }
    size_t get_total_samples() const override { return total_samples_.load(); }
    
    // Thread safety control
    void set_thread_safe(bool enable) override;
    bool is_thread_safe() const override { return thread_safe_; }

private:
    size_t get_random_index() const;
    void ensure_capacity_unlocked();
    
    template<typename LockType>
    void add_impl(ExperienceType experience);
};

// Implementation details in source file
extern template class UniformReplayBuffer<std::vector<float>, int>;
extern template class UniformReplayBuffer<std::vector<std::vector<float>>, std::vector<float>>;

} // namespace core
} // namespace replay_buffer