#include "../../include/core/simple_buffer.hpp"

namespace replay_buffer {
namespace core {

// Explicit template instantiations for common types
template class SimpleReplayBuffer<std::vector<float>, int>;
template class SimpleReplayBuffer<std::vector<std::vector<float>>, std::vector<float>>;

} // namespace core
} // namespace replay_buffer