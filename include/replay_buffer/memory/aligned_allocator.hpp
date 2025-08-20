#pragma once

#include <memory>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace replay_buffer {
namespace memory {

// Aligned memory allocator for SIMD optimization
template<typename T, size_t Alignment = 64>
class AlignedAllocator {
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
        using other = AlignedAllocator<U, Alignment>;
    };
    
    static constexpr size_t alignment = Alignment;
    
    AlignedAllocator() noexcept = default;
    
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    pointer allocate(size_type n) {
        if (n > max_size()) {
            throw std::bad_alloc();
        }
        
        size_type bytes = n * sizeof(T);
        void* ptr = nullptr;
        
#if defined(_WIN32)
        ptr = _aligned_malloc(bytes, Alignment);
        if (!ptr) throw std::bad_alloc();
#elif defined(__APPLE__) || defined(__linux__)
        if (posix_memalign(&ptr, Alignment, bytes) != 0) {
            throw std::bad_alloc();
        }
#else
        // Fallback for other systems
        ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr) throw std::bad_alloc();
#endif
        
        return static_cast<pointer>(ptr);
    }
    
    void deallocate(pointer p, size_type) noexcept {
        if (p) {
#if defined(_WIN32)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }
    }
    
    template<typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        new(p) U(std::forward<Args>(args)...);
    }
    
    template<typename U>
    void destroy(U* p) {
        p->~U();
    }
    
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
    
    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

// Aligned vector type for high-performance operations
template<typename T, size_t Alignment = 64>
using aligned_vector = std::vector<T, AlignedAllocator<T, Alignment>>;

// Cache-friendly memory layout utilities
template<size_t CacheLineSize = 64>
struct CacheAligned {
    static constexpr size_t cache_line_size = CacheLineSize;
    
    template<typename T>
    static size_t padded_size() {
        return ((sizeof(T) + CacheLineSize - 1) / CacheLineSize) * CacheLineSize;
    }
    
    template<typename T>
    static size_t elements_per_cache_line() {
        return CacheLineSize / sizeof(T);
    }
};

// Memory prefetching utilities for performance
class MemoryPrefetch {
public:
    enum class Locality {
        TEMPORAL_0,    // No temporal locality
        TEMPORAL_1,    // Low temporal locality
        TEMPORAL_2,    // Moderate temporal locality
        TEMPORAL_3     // High temporal locality
    };
    
    template<typename T>
    static void prefetch_read(const T* ptr, Locality locality = Locality::TEMPORAL_2) {
#if defined(__GNUC__) || defined(__clang__)
        int hint = static_cast<int>(locality);
        __builtin_prefetch(ptr, 0, hint); // 0 = read
#elif defined(_MSC_VER)
        _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0 + static_cast<int>(locality));
#else
        (void)ptr; (void)locality; // No-op on unsupported compilers
#endif
    }
    
    template<typename T>
    static void prefetch_write(T* ptr, Locality locality = Locality::TEMPORAL_2) {
#if defined(__GNUC__) || defined(__clang__)
        int hint = static_cast<int>(locality);
        __builtin_prefetch(ptr, 1, hint); // 1 = write
#elif defined(_MSC_VER)
        _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0 + static_cast<int>(locality));
#else
        (void)ptr; (void)locality; // No-op on unsupported compilers
#endif
    }
    
    // Prefetch range of memory
    template<typename Iterator>
    static void prefetch_range(Iterator begin, Iterator end, Locality locality = Locality::TEMPORAL_2) {
        constexpr size_t prefetch_distance = 64; // bytes
        
        auto current = begin;
        while (current != end) {
            prefetch_read(&(*current), locality);
            
            // Advance by cache line size
            size_t step = prefetch_distance / sizeof(typename Iterator::value_type);
            std::advance(current, std::min(step, static_cast<size_t>(std::distance(current, end))));
        }
    }
};

// NUMA-aware memory allocation (placeholder for future extension)
class NumaAllocator {
public:
    enum class Policy {
        LOCAL,      // Allocate on local NUMA node
        INTERLEAVED,// Interleave across all nodes
        PREFERRED   // Prefer specific node but allow fallback
    };
    
    template<typename T>
    static T* allocate_numa(size_t count, Policy policy = Policy::LOCAL, int preferred_node = -1) {
        // For now, use standard allocation
        // Future: integrate with libnuma or Windows NUMA APIs
        (void)policy; (void)preferred_node;
        return static_cast<T*>(std::aligned_alloc(64, count * sizeof(T)));
    }
    
    template<typename T>
    static void deallocate_numa(T* ptr) {
        std::free(ptr);
    }
    
    static int get_numa_node() {
        // Placeholder: return current NUMA node
        return 0;
    }
    
    static size_t get_numa_node_count() {
        // Placeholder: return number of NUMA nodes
        return 1;
    }
};

} // namespace memory
} // namespace replay_buffer