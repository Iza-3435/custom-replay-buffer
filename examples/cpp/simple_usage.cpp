#include "../../include/core/uniform_buffer.hpp"
#include "../../include/prioritization/prioritized_buffer.hpp"
#include <iostream>
#include <vector>
#include <random>

using namespace replay_buffer;

// Simple example demonstrating basic replay buffer usage
int main() {
    std::cout << "Custom Replay Buffer - Basic Usage Example\n";
    std::cout << "==========================================\n\n";
    
    // Example 1: Uniform Replay Buffer
    {
        std::cout << "Example 1: Uniform Replay Buffer\n";
        std::cout << "---------------------------------\n";
        
        // Create buffer configuration
        core::ReplayBufferConfig config;
        config.capacity = 1000;
        config.thread_safe = false; // Single-threaded for this example
        
        // Create uniform buffer for vector states and integer actions
        using StateType = std::vector<float>;
        using ActionType = int;
        using Experience = core::Experience<StateType, ActionType>;
        
        auto buffer = std::make_unique<core::UniformReplayBuffer<StateType, ActionType>>(
            config.capacity, config);
        
        // Generate some sample experiences
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> state_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> action_dist(0, 3);
        std::uniform_real_distribution<float> reward_dist(-10.0f, 10.0f);
        
        std::cout << "Adding experiences to buffer...\n";
        for (int i = 0; i < 100; ++i) {
            // Create random state
            StateType state(4); // 4-dimensional state
            StateType next_state(4);
            for (int j = 0; j < 4; ++j) {
                state[j] = state_dist(rng);
                next_state[j] = state_dist(rng);
            }
            
            Experience exp{
                std::move(state),
                action_dist(rng),
                reward_dist(rng),
                std::move(next_state),
                rng() % 20 == 0 // 5% chance of terminal state
            };
            
            buffer->add(std::move(exp));
        }
        
        std::cout << "Buffer size: " << buffer->size() << "/" << buffer->capacity() << "\n";
        
        // Sample a batch
        const size_t batch_size = 8;
        auto batch = buffer->sample(batch_size);
        
        std::cout << "Sampled batch of size: " << batch.size() << "\n";
        std::cout << "Sample rewards: ";
        for (size_t i = 0; i < std::min(batch.rewards.size(), size_t(5)); ++i) {
            std::cout << batch.rewards[i] << " ";
        }
        std::cout << "\n\n";
    }
    
    // Example 2: Prioritized Replay Buffer
    {
        std::cout << "Example 2: Prioritized Replay Buffer\n";
        std::cout << "------------------------------------\n";
        
        // Create prioritized buffer configuration
        core::ReplayBufferConfig config;
        config.capacity = 1000;
        config.thread_safe = false;
        config.enable_prioritization = true;
        config.alpha = 0.6; // Prioritization exponent
        config.beta = 0.4;  // Importance sampling exponent
        
        using StateType = std::vector<float>;
        using ActionType = int;
        
        auto buffer = std::make_unique<prioritization::PrioritizedReplayBuffer<StateType, ActionType>>(
            config.capacity, config);
        
        // Set custom priority function (based on absolute reward)
        buffer->set_priority_function([](const auto& exp) {
            return std::abs(exp.reward) + 0.1; // Higher reward magnitude = higher priority
        });
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> state_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> action_dist(0, 3);
        std::normal_distribution<float> reward_dist(0.0f, 5.0f); // Normal distribution for rewards
        
        std::cout << "Adding experiences with varying rewards...\n";
        for (int i = 0; i < 100; ++i) {
            StateType state(4);
            StateType next_state(4);
            for (int j = 0; j < 4; ++j) {
                state[j] = state_dist(rng);
                next_state[j] = state_dist(rng);
            }
            
            float reward = reward_dist(rng);
            if (i % 10 == 0) {
                reward *= 3.0f; // Make some rewards much larger
            }
            
            core::Experience<StateType, ActionType> exp{
                std::move(state),
                action_dist(rng),
                reward,
                std::move(next_state),
                rng() % 20 == 0
            };
            
            buffer->add(std::move(exp));
        }
        
        std::cout << "Buffer size: " << buffer->size() << "/" << buffer->capacity() << "\n";
        std::cout << "Average priority: " << buffer->get_avg_priority() << "\n";
        std::cout << "Total priority: " << buffer->get_total_priority() << "\n";
        
        // Sample a batch with importance weights
        const size_t batch_size = 8;
        auto batch = buffer->sample(batch_size);
        
        std::cout << "Sampled prioritized batch of size: " << batch.size() << "\n";
        std::cout << "Sample rewards and weights:\n";
        for (size_t i = 0; i < std::min(batch.rewards.size(), size_t(5)); ++i) {
            std::cout << "  Reward: " << std::fixed << std::setprecision(2) << batch.rewards[i] 
                      << ", Weight: " << batch.priorities[i] << "\n";
        }
        
        // Update priorities based on TD errors (simulated)
        std::vector<size_t> indices;
        std::vector<float> new_priorities;
        for (size_t i = 0; i < batch.indices.size(); ++i) {
            indices.push_back(batch.indices[i]);
            // Simulate TD error calculation
            float td_error = std::abs(batch.rewards[i]) + rng() % 100 / 100.0f;
            new_priorities.push_back(td_error);
        }
        
        buffer->update_priorities(indices, new_priorities);
        std::cout << "Updated priorities for sampled experiences\n\n";
    }
    
    // Example 3: Performance comparison
    {
        std::cout << "Example 3: Performance Comparison\n";
        std::cout << "--------------------------------\n";
        
        const size_t test_capacity = 10000;
        const size_t num_operations = 5000;
        const size_t batch_size = 32;
        
        using StateType = std::vector<float>;
        using ActionType = int;
        
        // Test uniform buffer
        auto uniform_buffer = std::make_unique<core::UniformReplayBuffer<StateType, ActionType>>(
            test_capacity, core::ReplayBufferConfig{});
        
        // Test prioritized buffer
        core::ReplayBufferConfig pri_config;
        pri_config.alpha = 0.6;
        pri_config.beta = 0.4;
        auto prioritized_buffer = std::make_unique<prioritization::PrioritizedReplayBuffer<StateType, ActionType>>(
            test_capacity, pri_config);
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> state_dist(-1.0f, 1.0f);
        std::uniform_int_distribution<int> action_dist(0, 3);
        std::uniform_real_distribution<float> reward_dist(-1.0f, 1.0f);
        
        // Fill buffers
        for (size_t i = 0; i < test_capacity; ++i) {
            StateType state(10);
            StateType next_state(10);
            for (int j = 0; j < 10; ++j) {
                state[j] = state_dist(rng);
                next_state[j] = state_dist(rng);
            }
            
            core::Experience<StateType, ActionType> exp1{
                state, action_dist(rng), reward_dist(rng), next_state, rng() % 100 == 0
            };
            core::Experience<StateType, ActionType> exp2 = exp1; // Copy for second buffer
            
            uniform_buffer->add(std::move(exp1));
            prioritized_buffer->add(std::move(exp2));
        }
        
        // Benchmark sampling
        auto benchmark_sampling = [&](const std::string& name, auto& buffer) {
            auto start = std::chrono::high_resolution_clock::now();
            
            for (size_t i = 0; i < num_operations; ++i) {
                auto batch = buffer->sample(batch_size);
                // Simulate some work with the batch
                volatile double sum = 0.0;
                for (const auto& reward : batch.rewards) {
                    sum += reward;
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double samples_per_second = (num_operations * 1e6) / duration.count();
            std::cout << name << " - Samples/sec: " << std::fixed << std::setprecision(0) 
                      << samples_per_second << "\n";
        };
        
        benchmark_sampling("Uniform Buffer     ", uniform_buffer);
        benchmark_sampling("Prioritized Buffer ", prioritized_buffer);
    }
    
    std::cout << "\nExample completed successfully!\n";
    return 0;
}