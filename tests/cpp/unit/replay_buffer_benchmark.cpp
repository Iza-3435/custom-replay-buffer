#include "../../include/core/uniform_buffer.hpp"
#include "../../include/prioritization/prioritized_buffer.hpp"
#include "custom_benchmark_framework.hpp"
#include <vector>
#include <random>
#include <chrono>

using namespace replay_buffer;

// Test data types
using VectorState = std::vector<float>;
using SimpleAction = int;
using TestExperience = core::Experience<VectorState, SimpleAction>;

// Generate random experience for testing
TestExperience generate_random_experience(std::mt19937& rng, size_t state_dim = 84) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> action_dist(0, 3);
    
    VectorState state(state_dim);
    VectorState next_state(state_dim);
    
    for (size_t i = 0; i < state_dim; ++i) {
        state[i] = dist(rng);
        next_state[i] = dist(rng);
    }
    
    return TestExperience{
        std::move(state),
        action_dist(rng),
        dist(rng),
        std::move(next_state),
        rng() % 10 == 0  // 10% terminal states
    };
}

// Benchmark uniform buffer insertion
void benchmark_uniform_buffer_insertion() {
    constexpr size_t capacity = 100000;
    constexpr size_t num_operations = 50000;
    
    core::ReplayBufferConfig config;
    config.thread_safe = false; // Single-threaded benchmark
    
    auto buffer = std::make_unique<core::UniformReplayBuffer<VectorState, SimpleAction>>(capacity, config);
    std::mt19937 rng(42);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_operations; ++i) {
        auto exp = generate_random_experience(rng);
        buffer->add(std::move(exp));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_second = (num_operations * 1e6) / duration.count();
    double avg_latency = duration.count() / static_cast<double>(num_operations);
    
    benchmark::print_result("UniformBuffer::add", ops_per_second, "ops/sec");
    benchmark::print_result("UniformBuffer::add_latency", avg_latency, "μs/op");
}

// Benchmark uniform buffer sampling
void benchmark_uniform_buffer_sampling() {
    constexpr size_t capacity = 100000;
    constexpr size_t batch_size = 32;
    constexpr size_t num_samples = 1000;
    
    core::ReplayBufferConfig config;
    config.thread_safe = false;
    
    auto buffer = std::make_unique<core::UniformReplayBuffer<VectorState, SimpleAction>>(capacity, config);
    std::mt19937 rng(42);
    
    // Fill buffer
    for (size_t i = 0; i < capacity; ++i) {
        auto exp = generate_random_experience(rng);
        buffer->add(std::move(exp));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_samples; ++i) {
        auto batch = buffer->sample(batch_size);
        benchmark::do_not_optimize(batch);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double samples_per_second = (num_samples * 1e6) / duration.count();
    double avg_latency = duration.count() / static_cast<double>(num_samples);
    
    benchmark::print_result("UniformBuffer::sample", samples_per_second, "samples/sec");
    benchmark::print_result("UniformBuffer::sample_latency", avg_latency, "μs/sample");
}

// Benchmark prioritized buffer insertion
void benchmark_prioritized_buffer_insertion() {
    constexpr size_t capacity = 100000;
    constexpr size_t num_operations = 50000;
    
    core::ReplayBufferConfig config;
    config.thread_safe = false;
    config.alpha = 0.6;
    config.beta = 0.4;
    
    auto buffer = std::make_unique<prioritization::PrioritizedReplayBuffer<VectorState, SimpleAction>>(capacity, config);
    std::mt19937 rng(42);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_operations; ++i) {
        auto exp = generate_random_experience(rng);
        buffer->add(std::move(exp));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double ops_per_second = (num_operations * 1e6) / duration.count();
    double avg_latency = duration.count() / static_cast<double>(num_operations);
    
    benchmark::print_result("PrioritizedBuffer::add", ops_per_second, "ops/sec");
    benchmark::print_result("PrioritizedBuffer::add_latency", avg_latency, "μs/op");
}

// Benchmark prioritized buffer sampling
void benchmark_prioritized_buffer_sampling() {
    constexpr size_t capacity = 100000;
    constexpr size_t batch_size = 32;
    constexpr size_t num_samples = 1000;
    
    core::ReplayBufferConfig config;
    config.thread_safe = false;
    config.alpha = 0.6;
    config.beta = 0.4;
    
    auto buffer = std::make_unique<prioritization::PrioritizedReplayBuffer<VectorState, SimpleAction>>(capacity, config);
    std::mt19937 rng(42);
    
    // Fill buffer
    for (size_t i = 0; i < capacity; ++i) {
        auto exp = generate_random_experience(rng);
        buffer->add(std::move(exp));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_samples; ++i) {
        auto batch = buffer->sample(batch_size);
        benchmark::do_not_optimize(batch);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double samples_per_second = (num_samples * 1e6) / duration.count();
    double avg_latency = duration.count() / static_cast<double>(num_samples);
    
    benchmark::print_result("PrioritizedBuffer::sample", samples_per_second, "samples/sec");
    benchmark::print_result("PrioritizedBuffer::sample_latency", avg_latency, "μs/sample");
}

// Memory usage benchmark
void benchmark_memory_usage() {
    constexpr size_t capacity = 100000;
    
    // Measure memory usage for different buffer types
    auto measure_memory = [](const std::string& name, auto& buffer, size_t capacity) {
        std::mt19937 rng(42);
        
        size_t initial_memory = benchmark::get_memory_usage();
        
        for (size_t i = 0; i < capacity; ++i) {
            auto exp = generate_random_experience(rng, 84); // Standard Atari state size
            buffer->add(std::move(exp));
        }
        
        size_t final_memory = benchmark::get_memory_usage();
        size_t memory_per_experience = (final_memory - initial_memory) / capacity;
        
        benchmark::print_result(name + "_memory_per_exp", memory_per_experience, "bytes");
    };
    
    // Uniform buffer
    {
        core::ReplayBufferConfig config;
        config.thread_safe = false;
        auto buffer = std::make_unique<core::UniformReplayBuffer<VectorState, SimpleAction>>(capacity, config);
        measure_memory("UniformBuffer", buffer, capacity);
    }
    
    // Prioritized buffer
    {
        core::ReplayBufferConfig config;
        config.thread_safe = false;
        auto buffer = std::make_unique<prioritization::PrioritizedReplayBuffer<VectorState, SimpleAction>>(capacity, config);
        measure_memory("PrioritizedBuffer", buffer, capacity);
    }
}

// Scalability benchmark
void benchmark_scalability() {
    std::vector<size_t> capacities = {1000, 10000, 100000, 1000000};
    constexpr size_t batch_size = 32;
    constexpr size_t num_samples = 100;
    
    for (size_t capacity : capacities) {
        core::ReplayBufferConfig config;
        config.thread_safe = false;
        
        auto buffer = std::make_unique<core::UniformReplayBuffer<VectorState, SimpleAction>>(capacity, config);
        std::mt19937 rng(42);
        
        // Fill buffer
        for (size_t i = 0; i < capacity; ++i) {
            auto exp = generate_random_experience(rng, 20); // Smaller state for speed
            buffer->add(std::move(exp));
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_samples; ++i) {
            auto batch = buffer->sample(batch_size);
            benchmark::do_not_optimize(batch);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_latency = duration.count() / static_cast<double>(num_samples);
        std::string name = "Scalability_" + std::to_string(capacity);
        benchmark::print_result(name, avg_latency, "μs/sample");
    }
}

// Thread safety benchmark
void benchmark_thread_safety() {
    constexpr size_t capacity = 100000;
    constexpr size_t operations_per_thread = 10000;
    constexpr size_t num_threads = 4;
    
    core::ReplayBufferConfig config;
    config.thread_safe = true;
    
    auto buffer = std::make_unique<core::UniformReplayBuffer<VectorState, SimpleAction>>(capacity, config);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    
    // Producer threads
    for (size_t t = 0; t < num_threads / 2; ++t) {
        threads.emplace_back([&buffer, operations_per_thread, t]() {
            std::mt19937 rng(42 + t);
            for (size_t i = 0; i < operations_per_thread; ++i) {
                auto exp = generate_random_experience(rng, 20);
                buffer->add(std::move(exp));
            }
        });
    }
    
    // Consumer threads
    for (size_t t = 0; t < num_threads / 2; ++t) {
        threads.emplace_back([&buffer, operations_per_thread]() {
            for (size_t i = 0; i < operations_per_thread / 32; ++i) {
                if (buffer->size() > 32) {
                    auto batch = buffer->sample(32);
                    benchmark::do_not_optimize(batch);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double total_ops = operations_per_thread * num_threads;
    double ops_per_second = (total_ops * 1000) / duration.count();
    
    benchmark::print_result("ThreadSafety_throughput", ops_per_second, "ops/sec");
}

int main() {
    benchmark::print_header("Replay Buffer Benchmarks");
    
    benchmark_uniform_buffer_insertion();
    benchmark_uniform_buffer_sampling();
    benchmark_prioritized_buffer_insertion();
    benchmark_prioritized_buffer_sampling();
    benchmark_memory_usage();
    benchmark_scalability();
    benchmark_thread_safety();
    
    benchmark::print_footer();
    return 0;
}