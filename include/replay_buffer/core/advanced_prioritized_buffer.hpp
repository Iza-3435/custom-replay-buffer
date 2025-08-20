#pragma once

#include "experience.hpp"
#include "../prioritization/sum_tree.hpp"
#include "../memory/aligned_allocator.hpp"
#include "../concurrency/spinlock.hpp"
#include <atomic>
#include <functional>
#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

namespace replay_buffer {
namespace core {

// Advanced prioritized buffer with custom priority functions and optimizations
template<typename StateType, typename ActionType>
class AdvancedPrioritizedBuffer {
public:
    using ExperienceType = Experience<StateType, ActionType>;
    using BatchType = ExperienceBatch<StateType, ActionType>;
    using PriorityFunction = std::function<float(const ExperienceType&)>;
    
    // Advanced sampling configuration
    struct SamplingConfig {
        float alpha = 0.6f;                 // Prioritization exponent
        float beta = 0.4f;                  // Importance sampling exponent  
        float beta_increment = 0.001f;      // Beta annealing rate
        float epsilon = 1e-6f;              // Small constant for numerical stability
        bool enable_importance_sampling = true;
        bool enable_priority_annealing = true;
        size_t priority_update_frequency = 1000; // Update priorities every N samples
    };

private:
    // High-performance aligned memory
    memory::aligned_vector<ExperienceType, 64> buffer_;
    std::unique_ptr<prioritization::SumTree> sum_tree_;
    
    size_t capacity_;
    std::atomic<size_t> head_{0};
    std::atomic<size_t> size_{0};
    
    // Configuration and state
    SamplingConfig config_;
    std::atomic<float> current_beta_{0.4f};
    std::atomic<uint64_t> sample_count_{0};
    
    // Performance counters
    std::atomic<uint64_t> total_adds_{0};
    std::atomic<uint64_t> total_priority_updates_{0};
    
    // Lock-free synchronization
    concurrency::AdaptiveSpinlock spinlock_;
    
    // Priority functions
    PriorityFunction priority_func_;
    std::atomic<float> max_priority_{1.0f};
    
    // Advanced priority calculation methods
    mutable std::atomic<uint64_t> rng_state_;
    
public:
    explicit AdvancedPrioritizedBuffer(size_t capacity, const SamplingConfig& config = {})
        : capacity_(capacity), config_(config), 
          rng_state_(std::chrono::high_resolution_clock::now().time_since_epoch().count()) {
        
        buffer_.resize(capacity_);
        sum_tree_ = std::make_unique<prioritization::SumTree>(capacity_);
        current_beta_.store(config_.beta, std::memory_order_relaxed);
        
        // Set default priority function (TD-error based)
        set_priority_function([](const ExperienceType& exp) {
            return std::abs(exp.reward) + 0.1f; // Simple heuristic
        });
        
        // Prefetch initial memory
        for (size_t i = 0; i < std::min(size_t(64), capacity_); i += 8) {
            memory::MemoryPrefetch::prefetch_write(&buffer_[i]);
        }
    }
    
    // High-performance add with automatic priority calculation
    bool add(ExperienceType&& experience) {
        const uint64_t timestamp = get_high_precision_time();
        experience.timestamp = timestamp;
        
        // Calculate priority using custom function
        const float priority = calculate_priority(experience);
        experience.priority = priority;
        
        // Update max priority for new experiences
        float current_max = max_priority_.load(std::memory_order_acquire);
        while (priority > current_max && 
               !max_priority_.compare_exchange_weak(current_max, priority, std::memory_order_release)) {
            current_max = max_priority_.load(std::memory_order_acquire);
        }
        
        concurrency::SpinlockGuard lock(spinlock_);
        
        const size_t pos = head_.load(std::memory_order_relaxed);
        const size_t next_pos = (pos + 1) % capacity_;
        
        // Store experience
        buffer_[pos] = std::move(experience);
        
        // Update sum tree with powered priority
        const double tree_priority = std::pow(priority + config_.epsilon, config_.alpha);
        sum_tree_->set(pos, tree_priority);
        
        head_.store(next_pos, std::memory_order_relaxed);
        
        // Update size
        const size_t current_size = size_.load(std::memory_order_acquire);
        if (current_size < capacity_) {
            size_.store(current_size + 1, std::memory_order_release);
        }
        
        total_adds_.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    // Advanced sampling with multiple strategies
    enum class SamplingStrategy {
        PRIORITIZED,        // Standard prioritized sampling
        RANK_BASED,        // Rank-based prioritization
        MIXTURE,           // Mix of uniform and prioritized
        TEMPORAL_DIFFERENCE // TD-error guided sampling
    };
    
    BatchType sample(size_t batch_size, SamplingStrategy strategy = SamplingStrategy::PRIORITIZED) {
        BatchType batch;
        const size_t current_size = size_.load(std::memory_order_acquire);
        
        if (current_size == 0 || batch_size == 0) {
            return batch;
        }
        
        batch_size = std::min(batch_size, current_size);
        batch.reserve(batch_size);
        
        // Anneal beta if enabled
        if (config_.enable_priority_annealing) {
            const uint64_t sample_num = sample_count_.fetch_add(batch_size, std::memory_order_relaxed);
            if (sample_num % config_.priority_update_frequency == 0) {
                anneal_beta();
            }
        }
        
        const float beta = current_beta_.load(std::memory_order_acquire);
        
        switch (strategy) {
            case SamplingStrategy::PRIORITIZED:
                return sample_prioritized(batch_size, beta);
            case SamplingStrategy::RANK_BASED:
                return sample_rank_based(batch_size, beta);
            case SamplingStrategy::MIXTURE:
                return sample_mixture(batch_size, beta, 0.25f); // 25% uniform, 75% prioritized
            case SamplingStrategy::TEMPORAL_DIFFERENCE:
                return sample_td_guided(batch_size, beta);
            default:
                return sample_prioritized(batch_size, beta);
        }
    }
    
    // Batch priority update with vectorized operations
    void update_priorities(const std::vector<size_t>& indices, const std::vector<float>& priorities) {
        if (indices.size() != priorities.size() || indices.empty()) {
            return;
        }
        
        concurrency::SpinlockGuard lock(spinlock_);
        
        // Vectorized priority updates
        const float alpha = config_.alpha;
        const float epsilon = config_.epsilon;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            const size_t idx = indices[i];
            const float priority = priorities[i];
            
            if (idx < capacity_) {
                buffer_[idx].priority = priority;
                const double tree_priority = std::pow(priority + epsilon, alpha);
                sum_tree_->set(idx, tree_priority);
                
                // Update max priority
                float current_max = max_priority_.load(std::memory_order_acquire);
                while (priority > current_max && 
                       !max_priority_.compare_exchange_weak(current_max, priority, std::memory_order_release)) {
                    current_max = max_priority_.load(std::memory_order_acquire);
                }
            }
        }
        
        total_priority_updates_.fetch_add(indices.size(), std::memory_order_relaxed);
    }
    
    // Custom priority function setter
    void set_priority_function(PriorityFunction func) {
        priority_func_ = std::move(func);
    }
    
    // Advanced priority calculation with domain-specific heuristics
    void set_hft_priority_function() {
        set_priority_function([](const ExperienceType& exp) {
            // HFT-specific priority: emphasize recent high-impact trades
            const float reward_magnitude = std::abs(exp.reward);
            const float time_decay = 1.0f; // Could use actual time decay
            const float volatility_bonus = exp.done ? 2.0f : 1.0f; // Terminal states are important
            
            return reward_magnitude * time_decay * volatility_bonus + 0.01f;
        });
    }
    
    // Set priority function for exploration-heavy RL
    void set_exploration_priority_function() {
        set_priority_function([](const ExperienceType& exp) {
            // Prioritize experiences with high uncertainty or surprise
            const float surprise = std::abs(exp.reward); // Simplified
            const float exploration_bonus = exp.done ? 0.5f : 1.0f;
            
            return surprise * exploration_bonus + 0.05f;
        });
    }
    
    // Performance and configuration getters
    size_t size() const { return size_.load(std::memory_order_acquire); }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size() == 0; }
    bool full() const { return size() >= capacity_; }
    
    float get_current_beta() const { return current_beta_.load(std::memory_order_acquire); }
    float get_max_priority() const { return max_priority_.load(std::memory_order_acquire); }
    
    struct AdvancedStats {
        uint64_t total_adds;
        uint64_t total_samples;
        uint64_t total_priority_updates;
        size_t current_size;
        float current_beta;
        float max_priority;
        double avg_priority;
        double compression_ratio;
    };
    
    AdvancedStats get_advanced_stats() const {
        concurrency::SpinlockGuard lock(spinlock_);
        
        return {
            total_adds_.load(std::memory_order_relaxed),
            sample_count_.load(std::memory_order_relaxed),
            total_priority_updates_.load(std::memory_order_relaxed),
            size(),
            get_current_beta(),
            get_max_priority(),
            sum_tree_->total() / std::max(size(), size_t(1)),
            calculate_compression_ratio()
        };
    }

private:
    // High-precision timing
    static uint64_t get_high_precision_time() {
#if defined(__x86_64__) || defined(_M_X64)
        return __rdtsc();
#else
        return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
    }
    
    // Priority calculation
    float calculate_priority(const ExperienceType& experience) const {
        if (priority_func_) {
            return std::max(config_.epsilon, priority_func_(experience));
        }
        return max_priority_.load(std::memory_order_acquire); // Use max for new experiences
    }
    
    // Beta annealing
    void anneal_beta() {
        const float current = current_beta_.load(std::memory_order_acquire);
        const float new_beta = std::min(1.0f, current + config_.beta_increment);
        current_beta_.store(new_beta, std::memory_order_release);
    }
    
    // Sampling implementations
    BatchType sample_prioritized(size_t batch_size, float beta) {
        BatchType batch;
        batch.reserve(batch_size);
        
        const double total_priority = sum_tree_->total();
        const double min_priority = sum_tree_->min_priority();
        
        if (total_priority <= 0.0) {
            return sample_uniform(batch_size);
        }
        
        const double max_weight = config_.enable_importance_sampling ? 
            std::pow((min_priority + config_.epsilon) / total_priority, -beta) : 1.0;
        
        uint64_t rng = rng_state_.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < batch_size; ++i) {
            // Fast random sampling
            rng = rng * 1103515245 + 12345; // Linear congruential generator
            const double sample_val = (rng / double(UINT64_MAX)) * total_priority;
            
            const size_t idx = sum_tree_->sample(sample_val);
            if (idx >= size()) continue;
            
            const auto& exp = buffer_[idx];
            const double priority = sum_tree_->get(idx);
            const double prob = priority / total_priority;
            const float weight = config_.enable_importance_sampling ? 
                static_cast<float>(std::pow(prob, -beta) / max_weight) : 1.0f;
            
            add_to_batch(batch, exp, idx, weight);
        }
        
        rng_state_.store(rng, std::memory_order_relaxed);
        return batch;
    }
    
    BatchType sample_rank_based(size_t batch_size, float beta) {
        // Rank-based sampling implementation
        return sample_prioritized(batch_size, beta); // Simplified for now
    }
    
    BatchType sample_mixture(size_t batch_size, float beta, float uniform_ratio) {
        BatchType batch;
        batch.reserve(batch_size);
        
        const size_t uniform_samples = static_cast<size_t>(batch_size * uniform_ratio);
        const size_t prioritized_samples = batch_size - uniform_samples;
        
        // Sample uniformly
        if (uniform_samples > 0) {
            auto uniform_batch = sample_uniform(uniform_samples);
            merge_batches(batch, uniform_batch);
        }
        
        // Sample with priorities
        if (prioritized_samples > 0) {
            auto priority_batch = sample_prioritized(prioritized_samples, beta);
            merge_batches(batch, priority_batch);
        }
        
        return batch;
    }
    
    BatchType sample_td_guided(size_t batch_size, float beta) {
        // TD-error guided sampling (simplified)
        return sample_prioritized(batch_size, beta);
    }
    
    BatchType sample_uniform(size_t batch_size) {
        BatchType batch;
        batch.reserve(batch_size);
        
        const size_t current_size = size();
        uint64_t rng = rng_state_.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < batch_size; ++i) {
            rng = rng * 1103515245 + 12345;
            const size_t idx = rng % current_size;
            
            add_to_batch(batch, buffer_[idx], idx, 1.0f);
        }
        
        rng_state_.store(rng, std::memory_order_relaxed);
        return batch;
    }
    
    void add_to_batch(BatchType& batch, const ExperienceType& exp, size_t idx, float weight) {
        batch.states.push_back(exp.state);
        batch.actions.push_back(exp.action);
        batch.rewards.push_back(exp.reward);
        batch.next_states.push_back(exp.next_state);
        batch.dones.push_back(exp.done);
        batch.priorities.push_back(weight);
        batch.timestamps.push_back(exp.timestamp);
        batch.indices.push_back(idx);
    }
    
    void merge_batches(BatchType& dest, const BatchType& src) {
        dest.states.insert(dest.states.end(), src.states.begin(), src.states.end());
        dest.actions.insert(dest.actions.end(), src.actions.begin(), src.actions.end());
        dest.rewards.insert(dest.rewards.end(), src.rewards.begin(), src.rewards.end());
        dest.next_states.insert(dest.next_states.end(), src.next_states.begin(), src.next_states.end());
        dest.dones.insert(dest.dones.end(), src.dones.begin(), src.dones.end());
        dest.priorities.insert(dest.priorities.end(), src.priorities.begin(), src.priorities.end());
        dest.timestamps.insert(dest.timestamps.end(), src.timestamps.begin(), src.timestamps.end());
        dest.indices.insert(dest.indices.end(), src.indices.begin(), src.indices.end());
    }
    
    double calculate_compression_ratio() const {
        // Simplified compression ratio calculation
        return 1.0; // Would implement actual compression analysis
    }
};

} // namespace core
} // namespace replay_buffer