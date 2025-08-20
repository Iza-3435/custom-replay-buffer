#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <atomic>

namespace replay_buffer {
namespace concurrency {

template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
    std::atomic<bool> shutdown_;

public:
    ThreadSafeQueue() : shutdown_(false) {}
    
    ~ThreadSafeQueue() {
        shutdown();
    }
    
    // Add element to the queue
    void push(const T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!shutdown_.load()) {
            queue_.push(item);
            condition_.notify_one();
        }
    }
    
    void push(T&& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!shutdown_.load()) {
            queue_.push(std::move(item));
            condition_.notify_one();
        }
    }
    
    // Try to pop element without blocking
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }
    
    // Pop element with blocking wait
    std::optional<T> wait_and_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { 
            return !queue_.empty() || shutdown_.load(); 
        });
        
        if (queue_.empty()) {
            return std::nullopt; // Shutdown was called
        }
        
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }
    
    // Pop element with timeout
    template<typename Rep, typename Period>
    std::optional<T> wait_for_pop(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!condition_.wait_for(lock, timeout, [this] { 
            return !queue_.empty() || shutdown_.load(); 
        })) {
            return std::nullopt; // Timeout
        }
        
        if (queue_.empty()) {
            return std::nullopt; // Shutdown was called
        }
        
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }
    
    // Check if queue is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    // Get queue size
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    // Clear all elements
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        queue_.swap(empty);
    }
    
    // Shutdown the queue (unblocks all waiting threads)
    void shutdown() {
        shutdown_.store(true);
        condition_.notify_all();
    }
    
    // Check if queue is shut down
    bool is_shutdown() const {
        return shutdown_.load();
    }
};

// Lock-free single producer, single consumer queue
template<typename T>
class SPSCQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    Node dummy_node_;

public:
    SPSCQueue() : head_(&dummy_node_), tail_(&dummy_node_) {}
    
    ~SPSCQueue() {
        while (Node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }
    
    // Producer side - only call from one thread
    void push(T item) {
        Node* new_node = new Node();
        T* data = new T(std::move(item));
        
        Node* prev_tail = tail_.exchange(new_node);
        prev_tail->data.store(data);
        prev_tail->next.store(new_node);
    }
    
    // Consumer side - only call from one thread
    std::optional<T> try_pop() {
        Node* head = head_.load();
        Node* next = head->next.load();
        
        if (next == nullptr) {
            return std::nullopt; // Queue is empty
        }
        
        T* data = next->data.load();
        if (data == nullptr) {
            return std::nullopt; // Data not ready yet
        }
        
        T result = *data;
        delete data;
        head_.store(next);
        delete head;
        
        return result;
    }
    
    // Check if queue is empty (approximate)
    bool empty() const {
        Node* head = head_.load();
        Node* next = head->next.load();
        return next == nullptr;
    }
};

} // namespace concurrency
} // namespace replay_buffer