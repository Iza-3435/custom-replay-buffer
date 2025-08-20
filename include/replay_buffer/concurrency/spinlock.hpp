#pragma once

#include <atomic>
#include <thread>
#include <chrono>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <immintrin.h>
    #define HAS_PAUSE_INSTRUCTION 1
#else
    #define HAS_PAUSE_INSTRUCTION 0
#endif

namespace replay_buffer {
namespace concurrency {

// Basic spinlock implementation
class Spinlock {
private:
    std::atomic<bool> locked_;
    
public:
    Spinlock() : locked_(false) {}
    
    void lock() {
        while (locked_.exchange(true, std::memory_order_acquire)) {
            // Spin with pause for better performance
            while (locked_.load(std::memory_order_relaxed)) {
                pause();
            }
        }
    }
    
    bool try_lock() {
        return !locked_.exchange(true, std::memory_order_acquire);
    }
    
    void unlock() {
        locked_.store(false, std::memory_order_release);
    }
    
private:
    static void pause() {
#if HAS_PAUSE_INSTRUCTION
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }
};

// Adaptive spinlock that yields after spinning for a while
class AdaptiveSpinlock {
private:
    std::atomic<bool> locked_;
    static constexpr int SPIN_LIMIT = 1000;
    
public:
    AdaptiveSpinlock() : locked_(false) {}
    
    void lock() {
        int spin_count = 0;
        
        while (locked_.exchange(true, std::memory_order_acquire)) {
            while (locked_.load(std::memory_order_relaxed)) {
                if (++spin_count < SPIN_LIMIT) {
                    pause();
                } else {
                    std::this_thread::yield();
                    spin_count = 0;
                }
            }
        }
    }
    
    bool try_lock() {
        return !locked_.exchange(true, std::memory_order_acquire);
    }
    
    void unlock() {
        locked_.store(false, std::memory_order_release);
    }
    
private:
    static void pause() {
#if HAS_PAUSE_INSTRUCTION
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }
};

// Ticket spinlock for fairness
class TicketSpinlock {
private:
    alignas(64) std::atomic<unsigned> now_serving_;
    alignas(64) std::atomic<unsigned> next_ticket_;
    
public:
    TicketSpinlock() : now_serving_(0), next_ticket_(0) {}
    
    void lock() {
        unsigned ticket = next_ticket_.fetch_add(1, std::memory_order_relaxed);
        
        while (now_serving_.load(std::memory_order_acquire) != ticket) {
            pause();
        }
    }
    
    bool try_lock() {
        unsigned serving = now_serving_.load(std::memory_order_acquire);
        unsigned ticket = next_ticket_.load(std::memory_order_acquire);
        
        if (serving != ticket) {
            return false;
        }
        
        return next_ticket_.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acquire);
    }
    
    void unlock() {
        now_serving_.fetch_add(1, std::memory_order_release);
    }
    
private:
    static void pause() {
#if HAS_PAUSE_INSTRUCTION
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }
};

// Read-Write spinlock
class RWSpinlock {
private:
    std::atomic<int> readers_;  // Positive = number of readers, -1 = writer
    
public:
    RWSpinlock() : readers_(0) {}
    
    void read_lock() {
        int expected;
        do {
            expected = readers_.load(std::memory_order_acquire);
            if (expected < 0) {
                pause();
                continue; // Writer present, wait
            }
        } while (!readers_.compare_exchange_weak(expected, expected + 1, 
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed));
    }
    
    bool try_read_lock() {
        int expected = readers_.load(std::memory_order_acquire);
        return expected >= 0 && 
               readers_.compare_exchange_strong(expected, expected + 1, std::memory_order_acquire);
    }
    
    void read_unlock() {
        readers_.fetch_sub(1, std::memory_order_release);
    }
    
    void write_lock() {
        int expected = 0;
        while (!readers_.compare_exchange_weak(expected, -1, 
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed)) {
            expected = 0;
            while (readers_.load(std::memory_order_relaxed) != 0) {
                pause();
            }
        }
    }
    
    bool try_write_lock() {
        int expected = 0;
        return readers_.compare_exchange_strong(expected, -1, std::memory_order_acquire);
    }
    
    void write_unlock() {
        readers_.store(0, std::memory_order_release);
    }
    
private:
    static void pause() {
#if HAS_PAUSE_INSTRUCTION
        _mm_pause();
#elif defined(__aarch64__)
        asm volatile("yield" ::: "memory");
#else
        std::this_thread::yield();
#endif
    }
};

// RAII lock guards for spinlocks
template<typename SpinlockType>
class SpinlockGuard {
private:
    SpinlockType& spinlock_;
    
public:
    explicit SpinlockGuard(SpinlockType& spinlock) : spinlock_(spinlock) {
        spinlock_.lock();
    }
    
    ~SpinlockGuard() {
        spinlock_.unlock();
    }
    
    // Non-copyable, non-movable
    SpinlockGuard(const SpinlockGuard&) = delete;
    SpinlockGuard& operator=(const SpinlockGuard&) = delete;
    SpinlockGuard(SpinlockGuard&&) = delete;
    SpinlockGuard& operator=(SpinlockGuard&&) = delete;
};

// Specialized guards for RWSpinlock
class ReadLockGuard {
private:
    RWSpinlock& rw_spinlock_;
    
public:
    explicit ReadLockGuard(RWSpinlock& rw_spinlock) : rw_spinlock_(rw_spinlock) {
        rw_spinlock_.read_lock();
    }
    
    ~ReadLockGuard() {
        rw_spinlock_.read_unlock();
    }
    
    ReadLockGuard(const ReadLockGuard&) = delete;
    ReadLockGuard& operator=(const ReadLockGuard&) = delete;
    ReadLockGuard(ReadLockGuard&&) = delete;
    ReadLockGuard& operator=(ReadLockGuard&&) = delete;
};

class WriteLockGuard {
private:
    RWSpinlock& rw_spinlock_;
    
public:
    explicit WriteLockGuard(RWSpinlock& rw_spinlock) : rw_spinlock_(rw_spinlock) {
        rw_spinlock_.write_lock();
    }
    
    ~WriteLockGuard() {
        rw_spinlock_.write_unlock();
    }
    
    WriteLockGuard(const WriteLockGuard&) = delete;
    WriteLockGuard& operator=(const WriteLockGuard&) = delete;
    WriteLockGuard(WriteLockGuard&&) = delete;
    WriteLockGuard& operator=(WriteLockGuard&&) = delete;
};

// Scoped timeout lock
template<typename SpinlockType>
class TimeoutSpinlockGuard {
private:
    SpinlockType& spinlock_;
    bool locked_;
    
public:
    template<typename Rep, typename Period>
    explicit TimeoutSpinlockGuard(SpinlockType& spinlock, 
                                 const std::chrono::duration<Rep, Period>& timeout)
        : spinlock_(spinlock), locked_(false) {
        
        auto start = std::chrono::steady_clock::now();
        while (!spinlock_.try_lock()) {
            if (std::chrono::steady_clock::now() - start > timeout) {
                return; // Timeout
            }
            std::this_thread::yield();
        }
        locked_ = true;
    }
    
    ~TimeoutSpinlockGuard() {
        if (locked_) {
            spinlock_.unlock();
        }
    }
    
    bool owns_lock() const { return locked_; }
    
    TimeoutSpinlockGuard(const TimeoutSpinlockGuard&) = delete;
    TimeoutSpinlockGuard& operator=(const TimeoutSpinlockGuard&) = delete;
    TimeoutSpinlockGuard(TimeoutSpinlockGuard&&) = delete;
    TimeoutSpinlockGuard& operator=(TimeoutSpinlockGuard&&) = delete;
};

} // namespace concurrency
} // namespace replay_buffer

#undef HAS_PAUSE_INSTRUCTION