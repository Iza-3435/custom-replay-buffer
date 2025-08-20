#pragma once

#include <atomic>
#include <memory>
#include <array>
#include <optional>

namespace replay_buffer {
namespace concurrency {

// Lock-free single-producer, single-consumer ring buffer
template<typename T, size_t Size>
class SPSCRingBuffer {
    static_assert(Size > 0 && (Size & (Size - 1)) == 0, "Size must be a power of 2");
    
private:
    static constexpr size_t mask_ = Size - 1;
    
    alignas(64) std::atomic<size_t> write_pos_;
    alignas(64) std::atomic<size_t> read_pos_;
    alignas(64) std::array<T, Size> buffer_;
    
public:
    SPSCRingBuffer() : write_pos_(0), read_pos_(0) {}
    
    // Producer side - only call from one thread
    bool push(const T& item) {
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) & mask_;
        
        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[current_write] = item;
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }
    
    bool push(T&& item) {
        size_t current_write = write_pos_.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) & mask_;
        
        if (next_write == read_pos_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[current_write] = std::move(item);
        write_pos_.store(next_write, std::memory_order_release);
        return true;
    }
    
    // Consumer side - only call from one thread
    std::optional<T> pop() {
        size_t current_read = read_pos_.load(std::memory_order_relaxed);
        
        if (current_read == write_pos_.load(std::memory_order_acquire)) {
            return std::nullopt; // Buffer empty
        }
        
        T item = std::move(buffer_[current_read]);
        read_pos_.store((current_read + 1) & mask_, std::memory_order_release);
        return item;
    }
    
    // Check if buffer is empty (approximate)
    bool empty() const {
        return read_pos_.load(std::memory_order_acquire) == 
               write_pos_.load(std::memory_order_acquire);
    }
    
    // Check if buffer is full (approximate)
    bool full() const {
        size_t write = write_pos_.load(std::memory_order_acquire);
        size_t read = read_pos_.load(std::memory_order_acquire);
        return ((write + 1) & mask_) == read;
    }
    
    // Get approximate size
    size_t size() const {
        size_t write = write_pos_.load(std::memory_order_acquire);
        size_t read = read_pos_.load(std::memory_order_acquire);
        return (write - read) & mask_;
    }
    
    static constexpr size_t capacity() { return Size; }
};

// Lock-free multiple-producer, multiple-consumer ring buffer
template<typename T, size_t Size>
class MPMCRingBuffer {
    static_assert(Size > 0 && (Size & (Size - 1)) == 0, "Size must be a power of 2");
    
private:
    struct Slot {
        alignas(64) std::atomic<size_t> sequence;
        T data;
        
        Slot() : sequence(0) {}
    };
    
    static constexpr size_t mask_ = Size - 1;
    
    alignas(64) std::atomic<size_t> enqueue_pos_;
    alignas(64) std::atomic<size_t> dequeue_pos_;
    alignas(64) std::array<Slot, Size> buffer_;
    
public:
    MPMCRingBuffer() : enqueue_pos_(0), dequeue_pos_(0) {
        for (size_t i = 0; i < Size; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }
    
    bool push(const T& item) {
        Slot* slot;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        
        for (;;) {
            slot = &buffer_[pos & mask_];
            size_t seq = slot->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Buffer full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        slot->data = item;
        slot->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }
    
    bool push(T&& item) {
        Slot* slot;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        
        for (;;) {
            slot = &buffer_[pos & mask_];
            size_t seq = slot->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
            
            if (diff == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false; // Buffer full
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        slot->data = std::move(item);
        slot->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }
    
    std::optional<T> pop() {
        Slot* slot;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        
        for (;;) {
            slot = &buffer_[pos & mask_];
            size_t seq = slot->sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
            
            if (diff == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return std::nullopt; // Buffer empty
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        
        T item = std::move(slot->data);
        slot->sequence.store(pos + mask_ + 1, std::memory_order_release);
        return item;
    }
    
    bool empty() const {
        size_t dequeue_pos = dequeue_pos_.load(std::memory_order_acquire);
        const Slot* slot = &buffer_[dequeue_pos & mask_];
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        return static_cast<intptr_t>(seq) - static_cast<intptr_t>(dequeue_pos + 1) < 0;
    }
    
    bool full() const {
        size_t enqueue_pos = enqueue_pos_.load(std::memory_order_acquire);
        const Slot* slot = &buffer_[enqueue_pos & mask_];
        size_t seq = slot->sequence.load(std::memory_order_acquire);
        return static_cast<intptr_t>(seq) - static_cast<intptr_t>(enqueue_pos) < 0;
    }
    
    size_t approximate_size() const {
        size_t enqueue = enqueue_pos_.load(std::memory_order_acquire);
        size_t dequeue = dequeue_pos_.load(std::memory_order_acquire);
        return enqueue - dequeue;
    }
    
    static constexpr size_t capacity() { return Size; }
};

// Dynamic lock-free ring buffer that can grow
template<typename T>
class DynamicLockFreeBuffer {
private:
    struct Node {
        alignas(64) std::atomic<Node*> next;
        alignas(64) std::atomic<size_t> sequence;
        T data;
        
        Node() : next(nullptr), sequence(0) {}
    };
    
    alignas(64) std::atomic<Node*> head_;
    alignas(64) std::atomic<Node*> tail_;
    alignas(64) std::atomic<size_t> size_;
    
    // Memory pool for node allocation
    std::atomic<Node*> free_list_;
    
    Node* allocate_node() {
        Node* node = free_list_.load(std::memory_order_acquire);
        while (node) {
            Node* next = node->next.load(std::memory_order_relaxed);
            if (free_list_.compare_exchange_weak(node, next, std::memory_order_release, std::memory_order_acquire)) {
                node->next.store(nullptr, std::memory_order_relaxed);
                return node;
            }
        }
        return new Node();
    }
    
    void deallocate_node(Node* node) {
        Node* head = free_list_.load(std::memory_order_relaxed);
        do {
            node->next.store(head, std::memory_order_relaxed);
        } while (!free_list_.compare_exchange_weak(head, node, std::memory_order_release, std::memory_order_relaxed));
    }

public:
    DynamicLockFreeBuffer() : head_(new Node()), tail_(head_.load()), size_(0), free_list_(nullptr) {
        head_.load()->sequence.store(0, std::memory_order_relaxed);
    }
    
    ~DynamicLockFreeBuffer() {
        // Clean up remaining nodes
        while (Node* head = head_.load()) {
            head_.store(head->next.load());
            delete head;
        }
        
        // Clean up free list
        while (Node* node = free_list_.load()) {
            free_list_.store(node->next.load());
            delete node;
        }
    }
    
    void push(const T& item) {
        Node* new_node = allocate_node();
        new_node->data = item;
        
        Node* prev_tail = tail_.exchange(new_node, std::memory_order_acq_rel);
        prev_tail->next.store(new_node, std::memory_order_release);
        
        size_.fetch_add(1, std::memory_order_relaxed);
    }
    
    void push(T&& item) {
        Node* new_node = allocate_node();
        new_node->data = std::move(item);
        
        Node* prev_tail = tail_.exchange(new_node, std::memory_order_acq_rel);
        prev_tail->next.store(new_node, std::memory_order_release);
        
        size_.fetch_add(1, std::memory_order_relaxed);
    }
    
    std::optional<T> pop() {
        Node* head = head_.load(std::memory_order_acquire);
        Node* next = head->next.load(std::memory_order_acquire);
        
        if (!next) {
            return std::nullopt; // Buffer empty
        }
        
        T item = std::move(next->data);
        if (head_.compare_exchange_strong(head, next, std::memory_order_release)) {
            deallocate_node(head);
            size_.fetch_sub(1, std::memory_order_relaxed);
            return item;
        }
        
        return std::nullopt; // Contention, try again
    }
    
    size_t size() const {
        return size_.load(std::memory_order_acquire);
    }
    
    bool empty() const {
        return size() == 0;
    }
};

} // namespace concurrency
} // namespace replay_buffer