#pragma once

#include <vector>
#include <cstdint>
#include <memory>

namespace replay_buffer {
namespace core {

template<typename StateType, typename ActionType>
struct Experience {
    StateType state;
    ActionType action;
    float reward;
    StateType next_state;
    bool done;
    float priority;
    uint64_t timestamp;

    Experience() = default;
    
    Experience(StateType s, ActionType a, float r, StateType ns, bool d, float p = 1.0f)
        : state(std::move(s)), action(std::move(a)), reward(r), 
          next_state(std::move(ns)), done(d), priority(p),
          timestamp(0) {}
};

// Common type aliases for different RL scenarios
using VectorExperience = Experience<std::vector<float>, int>;
using MatrixExperience = Experience<std::vector<std::vector<float>>, std::vector<float>>;

// Batch of experiences for efficient processing
template<typename StateType, typename ActionType>
struct ExperienceBatch {
    std::vector<StateType> states;
    std::vector<ActionType> actions;
    std::vector<float> rewards;
    std::vector<StateType> next_states;
    std::vector<bool> dones;
    std::vector<float> priorities;
    std::vector<uint64_t> timestamps;
    std::vector<size_t> indices; // Original buffer indices for priority updates

    size_t size() const { return states.size(); }
    
    void clear() {
        states.clear();
        actions.clear();
        rewards.clear();
        next_states.clear();
        dones.clear();
        priorities.clear();
        timestamps.clear();
        indices.clear();
    }

    void reserve(size_t capacity) {
        states.reserve(capacity);
        actions.reserve(capacity);
        rewards.reserve(capacity);
        next_states.reserve(capacity);
        dones.reserve(capacity);
        priorities.reserve(capacity);
        timestamps.reserve(capacity);
        indices.reserve(capacity);
    }
};

// Interface for custom experience serialization
template<typename ExperienceType>
class IExperienceSerializer {
public:
    virtual ~IExperienceSerializer() = default;
    virtual std::vector<uint8_t> serialize(const ExperienceType& exp) const = 0;
    virtual ExperienceType deserialize(const std::vector<uint8_t>& data) const = 0;
    virtual size_t serialized_size(const ExperienceType& exp) const = 0;
};

} // namespace core
} // namespace replay_buffer