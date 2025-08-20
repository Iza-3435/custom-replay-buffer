#include "../../include/core/simple_buffer.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace replay_buffer::core;

int main() {
    std::cout << "Custom Replay Buffer - Quick Test\n";
    std::cout << "==================================\n\n";

    // Create a simple buffer
    using StateType = std::vector<float>;
    using ActionType = int;
    using Experience = replay_buffer::core::Experience<StateType, ActionType>;
    
    SimpleReplayBuffer<StateType, ActionType> buffer(1000);
    
    std::cout << "Created buffer with capacity: " << buffer.capacity() << "\n";
    
    // Add some test experiences
    std::cout << "Adding test experiences...\n";
    for (int i = 0; i < 50; ++i) {
        StateType state = {static_cast<float>(i), static_cast<float>(i+1), static_cast<float>(i+2)};
        StateType next_state = {static_cast<float>(i+1), static_cast<float>(i+2), static_cast<float>(i+3)};
        
        Experience exp{
            std::move(state),
            i % 4,  // action
            static_cast<float>(i * 0.1), // reward
            std::move(next_state),
            i % 10 == 0  // done
        };
        
        buffer.add(std::move(exp));
    }
    
    std::cout << "Buffer size after adding: " << buffer.size() << "\n\n";
    
    // Sample a batch
    std::cout << "Sampling batch of size 8...\n";
    auto batch = buffer.sample(8);
    
    std::cout << "Sampled batch size: " << batch.size() << "\n";
    std::cout << "Sample data:\n";
    std::cout << "Index | Action | Reward | Done | State[0]\n";
    std::cout << "------|--------|--------|------|----------\n";
    
    for (size_t i = 0; i < batch.size(); ++i) {
        std::cout << std::setw(5) << batch.indices[i] 
                  << " | " << std::setw(6) << batch.actions[i]
                  << " | " << std::setw(6) << std::fixed << std::setprecision(1) << batch.rewards[i]
                  << " | " << std::setw(4) << (batch.dones[i] ? "Yes" : "No")
                  << " | " << std::setw(8) << batch.states[i][0] << "\n";
    }
    
    std::cout << "\nTest completed successfully! 🎉\n";
    
    // Performance test
    std::cout << "\nRunning quick performance test...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        auto batch = buffer.sample(32);
        // Simulate some work
        volatile double sum = 0.0;
        for (const auto& reward : batch.rewards) {
            sum += reward;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double samples_per_second = (1000.0 * 1e6) / duration.count();
    std::cout << "Performance: " << std::fixed << std::setprecision(0) 
              << samples_per_second << " samples/sec\n";
    
    return 0;
}