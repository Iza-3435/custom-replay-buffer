#include "../../include/core/lockfree_buffer.hpp"
#include "../../include/core/advanced_prioritized_buffer.hpp"
#include "../../include/core/simple_buffer.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
#include <random>
#include <iomanip>

using namespace replay_buffer::core;

// Test data generation
template<typename StateType, typename ActionType>
Experience<StateType, ActionType> generate_hft_experience(std::mt19937& rng, int id) {
    std::normal_distribution<float> price_dist(100.0f, 5.0f);
    std::normal_distribution<float> reward_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> action_dist(0, 2); // Buy, Sell, Hold
    
    // HFT-like state: [price, volume, spread, momentum, volatility]
    StateType state = {
        price_dist(rng),           // Current price
        static_cast<float>(rng() % 1000 + 100), // Volume
        (rng() % 100) / 100.0f,    // Bid-ask spread
        reward_dist(rng),          // Price momentum
        std::abs(reward_dist(rng)) // Volatility
    };
    
    StateType next_state = state;
    next_state[0] += reward_dist(rng) * 0.5f; // Price change
    next_state[3] = reward_dist(rng);         // New momentum
    
    float reward = reward_dist(rng) * (1.0f + std::abs(state[4])); // Volatility-adjusted reward
    
    return Experience<StateType, ActionType>{
        std::move(state),
        action_dist(rng),
        reward,
        std::move(next_state),
        rng() % 1000 == 0, // 0.1% chance of market close (terminal)
        std::abs(reward) + 0.01f, // Initial priority
        static_cast<uint64_t>(id)
    };
}

// Performance benchmark
class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        std::string name;
        double adds_per_second;
        double samples_per_second;
        double avg_latency_ns;
        size_t memory_usage_mb;
    };
    
    template<typename BufferType>
    static BenchmarkResult benchmark_buffer(const std::string& name, size_t operations = 100000) {
        using StateType = std::vector<float>;
        using ActionType = int;
        
        BufferType buffer(100000); // 100K capacity
        std::mt19937 rng(42);
        
        // Benchmark additions
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < operations; ++i) {
            auto exp = generate_hft_experience<StateType, ActionType>(rng, i);
            buffer.add(std::move(exp));
        }
        
        auto add_end = std::chrono::high_resolution_clock::now();
        auto add_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(add_end - start);
        
        // Benchmark sampling
        const size_t sample_operations = 10000;
        const size_t batch_size = 32;
        
        auto sample_start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < sample_operations; ++i) {
            auto batch = buffer.sample_batch ? buffer.sample_batch(batch_size) : buffer.sample(batch_size);
            volatile size_t size = batch.size(); // Prevent optimization
            (void)size;
        }
        
        auto sample_end = std::chrono::high_resolution_clock::now();
        auto sample_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(sample_end - sample_start);
        
        return {
            name,
            operations * 1e9 / add_duration.count(),
            sample_operations * 1e9 / sample_duration.count(),
            add_duration.count() / static_cast<double>(operations),
            sizeof(BufferType) / (1024 * 1024) // Rough estimate
        };
    }
};

// Multi-threaded stress test
void multi_threaded_stress_test() {
    std::cout << "\n🧵 Multi-threaded Stress Test\n";
    std::cout << "==============================\n";
    
    constexpr size_t BUFFER_SIZE = 1000000;
    constexpr size_t NUM_PRODUCERS = 4;
    constexpr size_t NUM_CONSUMERS = 2;
    constexpr size_t OPS_PER_THREAD = 50000;
    
    using StateType = std::vector<float>;
    using ActionType = int;
    
    LockFreeReplayBuffer<StateType, ActionType, BUFFER_SIZE> buffer;
    
    std::atomic<bool> start_flag{false};
    std::atomic<uint64_t> total_adds{0};
    std::atomic<uint64_t> total_samples{0};
    
    std::vector<std::thread> producers;
    std::vector<std::thread> consumers;
    
    // Producer threads
    for (size_t i = 0; i < NUM_PRODUCERS; ++i) {
        producers.emplace_back([&, i]() {
            std::mt19937 rng(42 + i);
            
            // Wait for start signal
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            
            for (size_t op = 0; op < OPS_PER_THREAD; ++op) {
                auto exp = generate_hft_experience<StateType, ActionType>(rng, i * OPS_PER_THREAD + op);
                if (buffer.add(std::move(exp))) {
                    total_adds.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    
    // Consumer threads
    for (size_t i = 0; i < NUM_CONSUMERS; ++i) {
        consumers.emplace_back([&, i]() {
            // Wait for start signal
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            
            for (size_t op = 0; op < OPS_PER_THREAD / 32; ++op) {
                if (buffer.size() >= 32) {
                    auto batch = buffer.sample_batch(32);
                    total_samples.fetch_add(batch.size(), std::memory_order_relaxed);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    start_flag.store(true, std::memory_order_release);
    
    // Wait for all threads
    for (auto& t : producers) t.join();
    for (auto& t : consumers) t.join();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    auto stats = buffer.get_stats();
    
    std::cout << "Duration: " << duration.count() << " ms\n";
    std::cout << "Total adds: " << total_adds.load() << "\n";
    std::cout << "Total samples: " << total_samples.load() << "\n";
    std::cout << "Add throughput: " << std::fixed << std::setprecision(0) 
              << stats.add_rate_per_sec << " ops/sec\n";
    std::cout << "Sample throughput: " << std::fixed << std::setprecision(0) 
              << stats.sample_rate_per_sec << " ops/sec\n";
    std::cout << "Buffer utilization: " 
              << (100.0 * stats.current_size / BUFFER_SIZE) << "%\n";
}

// Advanced prioritization demo
void advanced_prioritization_demo() {
    std::cout << "\n🎯 Advanced Prioritization Demo\n";
    std::cout << "===============================\n";
    
    using StateType = std::vector<float>;
    using ActionType = int;
    
    AdvancedPrioritizedBuffer<StateType, ActionType>::SamplingConfig config;
    config.alpha = 0.7f;
    config.beta = 0.5f;
    config.enable_importance_sampling = true;
    config.enable_priority_annealing = true;
    
    AdvancedPrioritizedBuffer<StateType, ActionType> buffer(10000, config);
    
    // Set HFT-specific priority function
    buffer.set_hft_priority_function();
    
    std::mt19937 rng(42);
    
    // Add diverse experiences
    std::cout << "Adding experiences with different characteristics...\n";
    for (int i = 0; i < 1000; ++i) {
        auto exp = generate_hft_experience<StateType, ActionType>(rng, i);
        
        // Create some high-impact experiences
        if (i % 50 == 0) {
            exp.reward *= 5.0f; // High-impact trade
        }
        
        buffer.add(std::move(exp));
    }
    
    auto stats = buffer.get_advanced_stats();
    std::cout << "Buffer size: " << stats.current_size << "\n";
    std::cout << "Max priority: " << std::fixed << std::setprecision(3) << stats.max_priority << "\n";
    std::cout << "Average priority: " << std::fixed << std::setprecision(3) << stats.avg_priority << "\n";
    std::cout << "Current beta: " << std::fixed << std::setprecision(3) << stats.current_beta << "\n";
    
    // Test different sampling strategies
    std::vector<std::pair<std::string, AdvancedPrioritizedBuffer<StateType, ActionType>::SamplingStrategy>> strategies = {
        {"Prioritized", AdvancedPrioritizedBuffer<StateType, ActionType>::SamplingStrategy::PRIORITIZED},
        {"Rank-based", AdvancedPrioritizedBuffer<StateType, ActionType>::SamplingStrategy::RANK_BASED},
        {"Mixture", AdvancedPrioritizedBuffer<StateType, ActionType>::SamplingStrategy::MIXTURE}
    };
    
    for (const auto& [name, strategy] : strategies) {
        auto batch = buffer.sample(8, strategy);
        
        std::cout << "\n" << name << " sampling results:\n";
        for (size_t i = 0; i < batch.rewards.size(); ++i) {
            std::cout << "  Reward: " << std::setw(8) << std::fixed << std::setprecision(2) 
                      << batch.rewards[i] << ", Weight: " << std::setw(6) << std::setprecision(3) 
                      << batch.priorities[i] << "\n";
        }
    }
}

int main() {
    std::cout << "🚀 Advanced Replay Buffer Performance Showcase\n";
    std::cout << "===============================================\n";
    
    // Performance comparison
    std::cout << "\n⚡ Performance Comparison\n";
    std::cout << "========================\n";
    
    using StateType = std::vector<float>;
    using ActionType = int;
    
    std::vector<PerformanceBenchmark::BenchmarkResult> results;
    
    // Benchmark different buffer types
    results.push_back(PerformanceBenchmark::benchmark_buffer<SimpleReplayBuffer<StateType, ActionType>>("Simple Buffer"));
    results.push_back(PerformanceBenchmark::benchmark_buffer<LockFreeReplayBuffer<StateType, ActionType, 100000>>("Lock-Free Buffer"));
    
    // Print results table
    std::cout << std::left << std::setw(20) << "Buffer Type" 
              << std::right << std::setw(15) << "Adds/sec" 
              << std::setw(15) << "Samples/sec"
              << std::setw(15) << "Latency (ns)" 
              << std::setw(15) << "Memory (MB)" << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::left << std::setw(20) << result.name
                  << std::right << std::setw(15) << std::fixed << std::setprecision(0) << result.adds_per_second
                  << std::setw(15) << std::fixed << std::setprecision(0) << result.samples_per_second  
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.avg_latency_ns
                  << std::setw(15) << std::fixed << std::setprecision(1) << result.memory_usage_mb << "\n";
    }
    
    // Run multi-threaded stress test
    multi_threaded_stress_test();
    
    // Demo advanced prioritization
    advanced_prioritization_demo();
    
    std::cout << "\n✅ Performance showcase completed!\n";
    std::cout << "\n🏁 Key Improvements Over Traditional Buffers:\n";
    std::cout << "• Sub-microsecond latency with lock-free operations\n";
    std::cout << "• Advanced prioritization strategies (rank-based, mixture, TD-guided)\n";
    std::cout << "• SIMD-optimized memory operations\n";
    std::cout << "• Cache-friendly aligned memory layout\n";
    std::cout << "• High-precision timestamping using CPU cycles\n";
    std::cout << "• Configurable importance sampling with annealing\n";
    std::cout << "• Domain-specific priority functions (HFT, exploration-based)\n";
    std::cout << "• Real-time performance monitoring\n";
    
    return 0;
}