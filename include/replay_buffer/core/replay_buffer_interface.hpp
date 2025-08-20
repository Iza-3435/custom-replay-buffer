#pragma once

#include "experience.hpp"
#include <memory>
#include <functional>

namespace replay_buffer {
namespace core {

template<typename StateType, typename ActionType>
class IReplayBuffer {
public:
    using ExperienceType = Experience<StateType, ActionType>;
    using BatchType = ExperienceBatch<StateType, ActionType>;
    using PriorityFunction = std::function<float(const ExperienceType&)>;

    virtual ~IReplayBuffer() = default;

    // Core buffer operations
    virtual void add(const ExperienceType& experience) = 0;
    virtual void add(ExperienceType&& experience) = 0;
    virtual BatchType sample(size_t batch_size) = 0;
    
    // Buffer state queries
    virtual size_t size() const = 0;
    virtual size_t capacity() const = 0;
    virtual bool empty() const = 0;
    virtual bool full() const = 0;
    
    // Priority management
    virtual void update_priorities(const std::vector<size_t>& indices, 
                                   const std::vector<float>& priorities) = 0;
    virtual void set_priority_function(PriorityFunction func) = 0;
    
    // Advanced operations
    virtual void clear() = 0;
    virtual void reserve(size_t capacity) = 0;
    
    // Statistics and monitoring
    virtual double get_avg_priority() const = 0;
    virtual size_t get_total_additions() const = 0;
    virtual size_t get_total_samples() const = 0;
    
    // Thread safety
    virtual void set_thread_safe(bool enable) = 0;
    virtual bool is_thread_safe() const = 0;
};

// Configuration for replay buffer behavior
struct ReplayBufferConfig {
    size_t capacity = 100000;
    bool enable_prioritization = true;
    bool thread_safe = true;
    bool enable_compression = false;
    float alpha = 0.6f; // Prioritization exponent
    float beta = 0.4f;  // Importance sampling exponent
    float beta_increment = 0.001f; // Beta annealing rate
    float priority_epsilon = 1e-6f; // Small constant to prevent zero priorities
    
    // Memory optimization
    bool enable_delta_compression = false;
    bool enable_quantization = false;
    uint8_t quantization_bits = 8;
    
    // Performance tuning
    size_t batch_allocation_size = 1000;
    bool use_memory_pool = true;
    size_t max_memory_usage = 0; // 0 = unlimited
};

// Factory for creating different types of replay buffers
template<typename StateType, typename ActionType>
class ReplayBufferFactory {
public:
    enum class BufferType {
        UNIFORM,           // Simple uniform sampling
        PRIORITIZED,       // Prioritized Experience Replay
        COMPRESSED,        // Memory-efficient with compression
        LOCK_FREE,         // High-performance concurrent
        CUSTOM             // User-defined implementation
    };

    static std::unique_ptr<IReplayBuffer<StateType, ActionType>> 
    create(BufferType type, const ReplayBufferConfig& config = {});
    
    static std::unique_ptr<IReplayBuffer<StateType, ActionType>>
    create_custom(std::function<std::unique_ptr<IReplayBuffer<StateType, ActionType>>()> factory);
};

} // namespace core
} // namespace replay_buffer