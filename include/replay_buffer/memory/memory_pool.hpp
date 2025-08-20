#pragma once

#include <memory>
#include <vector>
#include <stack>
#include <mutex>
#include <atomic>
#include <cstddef>

namespace replay_buffer {
namespace memory {

template<typename T>
class MemoryPool {
private:
    struct Block {
        alignas(T) char data[sizeof(T)];
        Block* next;
        
        Block() : next(nullptr) {}
    };
    
    std::vector<std::unique_ptr<Block[]>> chunks_;
    std::stack<Block*> free_blocks_;
    std::mutex mutex_;
    std::atomic<size_t> allocated_count_;
    std::atomic<size_t> total_capacity_;
    
    size_t chunk_size_;
    size_t blocks_per_chunk_;

public:
    explicit MemoryPool(size_t initial_capacity = 1000, size_t chunk_size = 1000)
        : allocated_count_(0), total_capacity_(0), 
          chunk_size_(chunk_size), blocks_per_chunk_(chunk_size) {
        allocate_chunk();
    }
    
    ~MemoryPool() {
        // Destructor will automatically clean up all chunks
    }
    
    // Allocate memory for one object
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (free_blocks_.empty()) {
            allocate_chunk();
        }
        
        Block* block = free_blocks_.top();
        free_blocks_.pop();
        allocated_count_.fetch_add(1, std::memory_order_relaxed);
        
        return reinterpret_cast<T*>(block->data);
    }
    
    // Deallocate memory
    void deallocate(T* ptr) {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        Block* block = reinterpret_cast<Block*>(
            reinterpret_cast<char*>(ptr) - offsetof(Block, data));
        
        free_blocks_.push(block);
        allocated_count_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    // Construct object in place
    template<typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        try {
            new(ptr) T(std::forward<Args>(args)...);
            return ptr;
        } catch (...) {
            deallocate(ptr);
            throw;
        }
    }
    
    // Destroy and deallocate object
    void destroy(T* ptr) {
        if (!ptr) return;
        ptr->~T();
        deallocate(ptr);
    }
    
    // Statistics
    size_t allocated_count() const {
        return allocated_count_.load(std::memory_order_relaxed);
    }
    
    size_t total_capacity() const {
        return total_capacity_.load(std::memory_order_relaxed);
    }
    
    size_t available_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_blocks_.size();
    }
    
    double utilization_ratio() const {
        size_t capacity = total_capacity();
        return capacity > 0 ? static_cast<double>(allocated_count()) / capacity : 0.0;
    }

private:
    void allocate_chunk() {
        auto chunk = std::make_unique<Block[]>(blocks_per_chunk_);
        Block* chunk_ptr = chunk.get();
        
        // Link all blocks in the chunk to the free list
        for (size_t i = 0; i < blocks_per_chunk_; ++i) {
            free_blocks_.push(&chunk_ptr[i]);
        }
        
        chunks_.push_back(std::move(chunk));
        total_capacity_.fetch_add(blocks_per_chunk_, std::memory_order_relaxed);
    }
};

// RAII wrapper for memory pool allocation
template<typename T>
class PoolAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = PoolAllocator<U>;
    };

private:
    std::shared_ptr<MemoryPool<T>> pool_;

public:
    explicit PoolAllocator(std::shared_ptr<MemoryPool<T>> pool)
        : pool_(std::move(pool)) {}
    
    template<typename U>
    PoolAllocator(const PoolAllocator<U>& other) : pool_(other.pool_) {}
    
    pointer allocate(size_type n) {
        if (n != 1) {
            throw std::bad_alloc(); // This allocator only supports single objects
        }
        return pool_->allocate();
    }
    
    void deallocate(pointer p, size_type n) {
        (void)n; // Unused parameter
        pool_->deallocate(p);
    }
    
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }
    
    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
    
    bool operator==(const PoolAllocator& other) const {
        return pool_ == other.pool_;
    }
    
    bool operator!=(const PoolAllocator& other) const {
        return !(*this == other);
    }
};

} // namespace memory
} // namespace replay_buffer