#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <functional>

namespace replay_buffer {
namespace prioritization {

// Sum tree data structure for efficient prioritized sampling
class SumTree {
private:
    std::vector<double> tree_;
    size_t capacity_;
    size_t data_start_; // Index where data nodes start
    
public:
    explicit SumTree(size_t capacity) 
        : capacity_(capacity), data_start_(capacity - 1) {
        // Tree size: internal nodes + leaf nodes
        // For capacity N, we need N-1 internal nodes + N leaf nodes = 2N-1 total
        tree_.resize(2 * capacity - 1, 0.0);
    }
    
    // Set priority for a specific index
    void set(size_t index, double priority) {
        if (index >= capacity_) return;
        
        size_t tree_index = data_start_ + index;
        double change = priority - tree_[tree_index];
        tree_[tree_index] = priority;
        
        // Propagate change up the tree
        while (tree_index > 0) {
            tree_index = (tree_index - 1) / 2;
            tree_[tree_index] += change;
        }
    }
    
    // Get priority for a specific index
    double get(size_t index) const {
        if (index >= capacity_) return 0.0;
        return tree_[data_start_ + index];
    }
    
    // Get total sum of all priorities
    double total() const {
        return tree_.empty() ? 0.0 : tree_[0];
    }
    
    // Sample an index based on priorities
    size_t sample(double value) const {
        if (total() <= 0.0) {
            // Fallback to uniform sampling
            return std::min(static_cast<size_t>(value * capacity_), capacity_ - 1);
        }
        
        size_t index = 0;
        
        // Navigate down the tree
        while (index < data_start_) {
            size_t left = 2 * index + 1;
            size_t right = 2 * index + 2;
            
            if (left >= tree_.size()) break;
            
            double left_sum = tree_[left];
            
            if (value <= left_sum) {
                index = left;
            } else {
                value -= left_sum;
                index = right;
            }
        }
        
        // Convert tree index to data index
        return index - data_start_;
    }
    
    // Sample an index with uniform random value
    template<typename RNG>
    size_t sample(RNG& rng) const {
        std::uniform_real_distribution<double> dist(0.0, total());
        return sample(dist(rng));
    }
    
    // Update multiple priorities efficiently
    void update_batch(const std::vector<size_t>& indices, 
                     const std::vector<double>& priorities) {
        if (indices.size() != priorities.size()) return;
        
        for (size_t i = 0; i < indices.size(); ++i) {
            set(indices[i], priorities[i]);
        }
    }
    
    // Get the minimum priority in the tree
    double min_priority() const {
        double min_val = std::numeric_limits<double>::max();
        for (size_t i = 0; i < capacity_; ++i) {
            min_val = std::min(min_val, get(i));
        }
        return min_val;
    }
    
    // Get the maximum priority in the tree
    double max_priority() const {
        double max_val = 0.0;
        for (size_t i = 0; i < capacity_; ++i) {
            max_val = std::max(max_val, get(i));
        }
        return max_val;
    }
    
    size_t capacity() const { return capacity_; }
};

// Segment tree for range queries and updates
class SegmentTree {
private:
    std::vector<double> tree_;
    size_t n_;
    std::function<double(double, double)> combine_func_;
    double identity_;
    
public:
    SegmentTree(size_t size, std::function<double(double, double)> combine_func, double identity)
        : n_(1), combine_func_(combine_func), identity_(identity) {
        
        // Find next power of 2
        while (n_ < size) n_ *= 2;
        tree_.resize(2 * n_, identity);
    }
    
    void update(size_t pos, double value) {
        pos += n_;
        tree_[pos] = value;
        
        while (pos > 1) {
            pos /= 2;
            tree_[pos] = combine_func_(tree_[2 * pos], tree_[2 * pos + 1]);
        }
    }
    
    double query(size_t left, size_t right) const {
        double result = identity_;
        left += n_;
        right += n_;
        
        while (left < right) {
            if (left % 2 == 1) {
                result = combine_func_(result, tree_[left++]);
            }
            if (right % 2 == 1) {
                result = combine_func_(result, tree_[--right]);
            }
            left /= 2;
            right /= 2;
        }
        
        return result;
    }
    
    double query_all() const {
        return tree_[1];
    }
    
    size_t size() const { return n_; }
};

// Priority sampler using sum tree for efficient weighted sampling
template<typename T>
class PrioritySampler {
private:
    std::vector<T> data_;
    SumTree sum_tree_;
    std::mt19937 rng_;
    double alpha_; // Prioritization exponent
    double beta_;  // Importance sampling exponent
    double epsilon_; // Small constant to prevent zero priorities
    
public:
    PrioritySampler(size_t capacity, double alpha = 0.6, double beta = 0.4, double epsilon = 1e-6)
        : sum_tree_(capacity), rng_(std::random_device{}()), 
          alpha_(alpha), beta_(beta), epsilon_(epsilon) {
        data_.reserve(capacity);
    }
    
    void add(const T& item, double priority) {
        if (data_.size() < sum_tree_.capacity()) {
            data_.push_back(item);
            sum_tree_.set(data_.size() - 1, std::pow(priority + epsilon_, alpha_));
        } else {
            // Replace oldest item (circular buffer behavior)
            static size_t next_index = 0;
            data_[next_index] = item;
            sum_tree_.set(next_index, std::pow(priority + epsilon_, alpha_));
            next_index = (next_index + 1) % sum_tree_.capacity();
        }
    }
    
    void add(T&& item, double priority) {
        if (data_.size() < sum_tree_.capacity()) {
            data_.push_back(std::move(item));
            sum_tree_.set(data_.size() - 1, std::pow(priority + epsilon_, alpha_));
        } else {
            static size_t next_index = 0;
            data_[next_index] = std::move(item);
            sum_tree_.set(next_index, std::pow(priority + epsilon_, alpha_));
            next_index = (next_index + 1) % sum_tree_.capacity();
        }
    }
    
    // Sample a single item
    std::pair<size_t, T> sample() {
        if (data_.empty()) {
            throw std::runtime_error("Cannot sample from empty priority sampler");
        }
        
        size_t index = sum_tree_.sample(rng_);
        index = std::min(index, data_.size() - 1);
        return {index, data_[index]};
    }
    
    // Sample multiple items with importance sampling weights
    struct SampledItem {
        size_t index;
        T item;
        double weight; // Importance sampling weight
    };
    
    std::vector<SampledItem> sample_batch(size_t batch_size) {
        std::vector<SampledItem> result;
        result.reserve(batch_size);
        
        if (data_.empty()) return result;
        
        double total_priority = sum_tree_.total();
        double min_priority = sum_tree_.min_priority();
        double max_weight = std::pow((min_priority + epsilon_) / total_priority, -beta_);
        
        for (size_t i = 0; i < batch_size; ++i) {
            size_t index = sum_tree_.sample(rng_);
            index = std::min(index, data_.size() - 1);
            
            double priority = sum_tree_.get(index);
            double prob = priority / total_priority;
            double weight = std::pow(prob, -beta_) / max_weight;
            
            result.push_back({index, data_[index], weight});
        }
        
        return result;
    }
    
    void update_priority(size_t index, double priority) {
        if (index < data_.size()) {
            sum_tree_.set(index, std::pow(priority + epsilon_, alpha_));
        }
    }
    
    void update_priorities(const std::vector<size_t>& indices, 
                          const std::vector<double>& priorities) {
        if (indices.size() != priorities.size()) return;
        
        std::vector<double> powered_priorities;
        powered_priorities.reserve(priorities.size());
        
        for (double priority : priorities) {
            powered_priorities.push_back(std::pow(priority + epsilon_, alpha_));
        }
        
        sum_tree_.update_batch(indices, powered_priorities);
    }
    
    size_t size() const { return data_.size(); }
    size_t capacity() const { return sum_tree_.capacity(); }
    
    double get_alpha() const { return alpha_; }
    double get_beta() const { return beta_; }
    
    void set_alpha(double alpha) { alpha_ = alpha; }
    void set_beta(double beta) { beta_ = beta; }
    
    double total_priority() const { return sum_tree_.total(); }
    double min_priority() const { return sum_tree_.min_priority(); }
    double max_priority() const { return sum_tree_.max_priority(); }
};

} // namespace prioritization
} // namespace replay_buffer