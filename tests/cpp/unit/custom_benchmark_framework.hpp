#pragma once

#include <iostream>
#include <string>
#include <chrono>
#include <iomanip>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>

#if defined(__linux__)
    #include <sys/resource.h>
    #include <unistd.h>
#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <sys/resource.h>
#elif defined(_WIN32)
    #include <windows.h>
    #include <psapi.h>
#endif

namespace benchmark {

// Prevent compiler optimizations from removing code
template<typename T>
void do_not_optimize(const T& value) {
    volatile auto* ptr = &value;
    (void)ptr;
}

// Get current memory usage in bytes
size_t get_memory_usage() {
#if defined(__linux__)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024; // Linux reports in KB
#elif defined(__APPLE__)
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // macOS reports in bytes
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc));
    return pmc.WorkingSetSize;
#else
    return 0; // Unsupported platform
#endif
}

// Simple timer class
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
    
    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return static_cast<double>(duration.count());
    }
    
    double elapsed_ns() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_);
        return static_cast<double>(duration.count());
    }
};

// Benchmark runner with warmup and multiple iterations
template<typename Func>
double run_benchmark(const std::string& name, Func&& func, size_t iterations = 10, size_t warmup = 3) {
    // Warmup runs
    for (size_t i = 0; i < warmup; ++i) {
        func();
    }
    
    Timer timer;
    
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    
    return timer.elapsed_us() / iterations;
}

// Result formatting and printing
void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << std::left << std::setw(30) << "Benchmark" 
              << std::right << std::setw(15) << "Result"
              << std::setw(15) << "Unit" << "\n";
    std::cout << std::string(60, '-') << "\n";
}

void print_result(const std::string& name, double value, const std::string& unit) {
    std::cout << std::left << std::setw(30) << name 
              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << value
              << std::setw(15) << unit << "\n";
}

void print_footer() {
    std::cout << std::string(60, '=') << "\n\n";
}

// CPU information
struct CPUInfo {
    int cores;
    int logical_cores;
    std::string model;
    
    static CPUInfo get() {
        CPUInfo info;
        info.cores = std::thread::hardware_concurrency();
        info.logical_cores = info.cores; // Simplified
        info.model = "Unknown";
        
#if defined(__linux__) || defined(__APPLE__)
        // Try to read CPU model from /proc/cpuinfo or similar
        // Simplified implementation
        info.model = "Generic CPU";
#elif defined(_WIN32)
        info.model = "Windows CPU";
#endif
        
        return info;
    }
};

// Memory information
struct MemoryInfo {
    size_t total_physical;
    size_t available_physical;
    
    static MemoryInfo get() {
        MemoryInfo info;
        info.total_physical = 0;
        info.available_physical = 0;
        
#if defined(__linux__)
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        info.total_physical = pages * page_size;
        
        long avail_pages = sysconf(_SC_AVPHYS_PAGES);
        info.available_physical = avail_pages * page_size;
#elif defined(__APPLE__)
        // Simplified implementation for macOS
        info.total_physical = 8ULL * 1024 * 1024 * 1024; // Assume 8GB
        info.available_physical = info.total_physical / 2;
#elif defined(_WIN32)
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        GlobalMemoryStatusEx(&statex);
        info.total_physical = statex.ullTotalPhys;
        info.available_physical = statex.ullAvailPhys;
#endif
        
        return info;
    }
};

// System information printer
void print_system_info() {
    auto cpu = CPUInfo::get();
    auto memory = MemoryInfo::get();
    
    std::cout << "System Information:\n";
    std::cout << "  CPU: " << cpu.model << " (" << cpu.cores << " cores)\n";
    std::cout << "  Memory: " << (memory.total_physical / (1024 * 1024)) << " MB total, "
              << (memory.available_physical / (1024 * 1024)) << " MB available\n";
    std::cout << "\n";
}

// Latency distribution tracker
class LatencyTracker {
private:
    std::vector<double> samples_;
    
public:
    void add_sample(double latency_us) {
        samples_.push_back(latency_us);
    }
    
    void print_statistics(const std::string& name) const {
        if (samples_.empty()) return;
        
        auto sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        
        double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
        double p50 = sorted[sorted.size() * 0.5];
        double p95 = sorted[sorted.size() * 0.95];
        double p99 = sorted[sorted.size() * 0.99];
        double min_val = sorted.front();
        double max_val = sorted.back();
        
        std::cout << name << " Latency Distribution:\n";
        print_result("  Mean", mean, "μs");
        print_result("  P50", p50, "μs");
        print_result("  P95", p95, "μs");
        print_result("  P99", p99, "μs");
        print_result("  Min", min_val, "μs");
        print_result("  Max", max_val, "μs");
    }
    
    void clear() {
        samples_.clear();
    }
};

// Throughput measurement helper
class ThroughputMeasurer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    size_t operations_;
    
public:
    ThroughputMeasurer() : start_(std::chrono::high_resolution_clock::now()), operations_(0) {}
    
    void record_operation() {
        ++operations_;
    }
    
    double get_ops_per_second() const {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_);
        return (operations_ * 1e6) / duration.count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
        operations_ = 0;
    }
};

} // namespace benchmark