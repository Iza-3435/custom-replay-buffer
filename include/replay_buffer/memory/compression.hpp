#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <cmath>

namespace replay_buffer {
namespace memory {

// Abstract base class for compression algorithms
template<typename T>
class ICompressor {
public:
    virtual ~ICompressor() = default;
    virtual std::vector<uint8_t> compress(const T& data) const = 0;
    virtual T decompress(const std::vector<uint8_t>& compressed) const = 0;
    virtual size_t compressed_size(const T& data) const = 0;
    virtual double compression_ratio(const T& data) const = 0;
};

// Delta compression for sequential data
template<typename T>
class DeltaCompressor : public ICompressor<std::vector<T>> {
private:
    T base_value_;
    
public:
    explicit DeltaCompressor(T base_value = T{}) : base_value_(base_value) {}
    
    std::vector<uint8_t> compress(const std::vector<T>& data) const override {
        if (data.empty()) return {};
        
        std::vector<uint8_t> result;
        result.reserve(data.size() * sizeof(T));
        
        // Store first value as base
        const uint8_t* base_ptr = reinterpret_cast<const uint8_t*>(&data[0]);
        result.insert(result.end(), base_ptr, base_ptr + sizeof(T));
        
        // Store deltas
        T prev = data[0];
        for (size_t i = 1; i < data.size(); ++i) {
            T delta = data[i] - prev;
            const uint8_t* delta_ptr = reinterpret_cast<const uint8_t*>(&delta);
            result.insert(result.end(), delta_ptr, delta_ptr + sizeof(T));
            prev = data[i];
        }
        
        return result;
    }
    
    std::vector<T> decompress(const std::vector<uint8_t>& compressed) const override {
        if (compressed.empty()) return {};
        
        size_t num_elements = compressed.size() / sizeof(T);
        std::vector<T> result;
        result.reserve(num_elements);
        
        // Read base value
        T current = *reinterpret_cast<const T*>(compressed.data());
        result.push_back(current);
        
        // Apply deltas
        for (size_t i = 1; i < num_elements; ++i) {
            T delta = *reinterpret_cast<const T*>(compressed.data() + i * sizeof(T));
            current += delta;
            result.push_back(current);
        }
        
        return result;
    }
    
    size_t compressed_size(const std::vector<T>& data) const override {
        return data.size() * sizeof(T); // Same size, but better compression with entropy coding
    }
    
    double compression_ratio(const std::vector<T>& data) const override {
        return 1.0; // Delta compression mainly helps with entropy coding
    }
};

// Quantization compressor for floating-point data
class QuantizationCompressor : public ICompressor<std::vector<float>> {
private:
    uint8_t bits_;
    float min_val_;
    float max_val_;
    
public:
    explicit QuantizationCompressor(uint8_t bits = 8, float min_val = -1.0f, float max_val = 1.0f)
        : bits_(bits), min_val_(min_val), max_val_(max_val) {}
    
    std::vector<uint8_t> compress(const std::vector<float>& data) const override {
        if (data.empty()) return {};
        
        uint32_t levels = (1U << bits_) - 1;
        float scale = levels / (max_val_ - min_val_);
        
        std::vector<uint8_t> result;
        
        // Store header: bits, min_val, max_val, size
        result.push_back(bits_);
        const uint8_t* min_ptr = reinterpret_cast<const uint8_t*>(&min_val_);
        const uint8_t* max_ptr = reinterpret_cast<const uint8_t*>(&max_val_);
        result.insert(result.end(), min_ptr, min_ptr + sizeof(float));
        result.insert(result.end(), max_ptr, max_ptr + sizeof(float));
        
        uint32_t size = static_cast<uint32_t>(data.size());
        const uint8_t* size_ptr = reinterpret_cast<const uint8_t*>(&size);
        result.insert(result.end(), size_ptr, size_ptr + sizeof(uint32_t));
        
        // Quantize and pack data
        if (bits_ <= 8) {
            for (float val : data) {
                uint8_t quantized = static_cast<uint8_t>(
                    std::clamp((val - min_val_) * scale, 0.0f, static_cast<float>(levels)));
                result.push_back(quantized);
            }
        } else if (bits_ <= 16) {
            for (float val : data) {
                uint16_t quantized = static_cast<uint16_t>(
                    std::clamp((val - min_val_) * scale, 0.0f, static_cast<float>(levels)));
                const uint8_t* q_ptr = reinterpret_cast<const uint8_t*>(&quantized);
                result.insert(result.end(), q_ptr, q_ptr + sizeof(uint16_t));
            }
        }
        
        return result;
    }
    
    std::vector<float> decompress(const std::vector<uint8_t>& compressed) const override {
        if (compressed.size() < 1 + 2 * sizeof(float) + sizeof(uint32_t)) {
            return {};
        }
        
        size_t offset = 0;
        uint8_t bits = compressed[offset++];
        
        float min_val = *reinterpret_cast<const float*>(compressed.data() + offset);
        offset += sizeof(float);
        
        float max_val = *reinterpret_cast<const float*>(compressed.data() + offset);
        offset += sizeof(float);
        
        uint32_t size = *reinterpret_cast<const uint32_t*>(compressed.data() + offset);
        offset += sizeof(uint32_t);
        
        uint32_t levels = (1U << bits) - 1;
        float scale = (max_val - min_val) / levels;
        
        std::vector<float> result;
        result.reserve(size);
        
        if (bits <= 8) {
            for (uint32_t i = 0; i < size; ++i) {
                uint8_t quantized = compressed[offset + i];
                result.push_back(min_val + quantized * scale);
            }
        } else if (bits <= 16) {
            for (uint32_t i = 0; i < size; ++i) {
                uint16_t quantized = *reinterpret_cast<const uint16_t*>(
                    compressed.data() + offset + i * sizeof(uint16_t));
                result.push_back(min_val + quantized * scale);
            }
        }
        
        return result;
    }
    
    size_t compressed_size(const std::vector<float>& data) const override {
        size_t header_size = 1 + 2 * sizeof(float) + sizeof(uint32_t);
        size_t data_size = data.size() * ((bits_ + 7) / 8); // Round up to nearest byte
        return header_size + data_size;
    }
    
    double compression_ratio(const std::vector<float>& data) const override {
        size_t original_size = data.size() * sizeof(float);
        size_t compressed = compressed_size(data);
        return static_cast<double>(original_size) / compressed;
    }
};

// Run-length encoding for sparse data
template<typename T>
class RLECompressor : public ICompressor<std::vector<T>> {
public:
    std::vector<uint8_t> compress(const std::vector<T>& data) const override {
        if (data.empty()) return {};
        
        std::vector<uint8_t> result;
        
        T current = data[0];
        uint32_t count = 1;
        
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] == current && count < UINT32_MAX) {
                count++;
            } else {
                // Write current run
                const uint8_t* val_ptr = reinterpret_cast<const uint8_t*>(&current);
                const uint8_t* count_ptr = reinterpret_cast<const uint8_t*>(&count);
                result.insert(result.end(), val_ptr, val_ptr + sizeof(T));
                result.insert(result.end(), count_ptr, count_ptr + sizeof(uint32_t));
                
                current = data[i];
                count = 1;
            }
        }
        
        // Write final run
        const uint8_t* val_ptr = reinterpret_cast<const uint8_t*>(&current);
        const uint8_t* count_ptr = reinterpret_cast<const uint8_t*>(&count);
        result.insert(result.end(), val_ptr, val_ptr + sizeof(T));
        result.insert(result.end(), count_ptr, count_ptr + sizeof(uint32_t));
        
        return result;
    }
    
    std::vector<T> decompress(const std::vector<uint8_t>& compressed) const override {
        std::vector<T> result;
        
        size_t offset = 0;
        size_t run_size = sizeof(T) + sizeof(uint32_t);
        
        while (offset + run_size <= compressed.size()) {
            T value = *reinterpret_cast<const T*>(compressed.data() + offset);
            offset += sizeof(T);
            
            uint32_t count = *reinterpret_cast<const uint32_t*>(compressed.data() + offset);
            offset += sizeof(uint32_t);
            
            for (uint32_t i = 0; i < count; ++i) {
                result.push_back(value);
            }
        }
        
        return result;
    }
    
    size_t compressed_size(const std::vector<T>& data) const override {
        // Worst case: no compression (every element different)
        return data.size() * (sizeof(T) + sizeof(uint32_t));
    }
    
    double compression_ratio(const std::vector<T>& data) const override {
        // Estimate based on run analysis
        if (data.empty()) return 1.0;
        
        size_t runs = 1;
        for (size_t i = 1; i < data.size(); ++i) {
            if (data[i] != data[i-1]) runs++;
        }
        
        size_t original_size = data.size() * sizeof(T);
        size_t compressed = runs * (sizeof(T) + sizeof(uint32_t));
        return static_cast<double>(original_size) / compressed;
    }
};

// Composite compressor that applies multiple compression techniques
template<typename T>
class CompositeCompressor : public ICompressor<T> {
private:
    std::vector<std::unique_ptr<ICompressor<T>>> compressors_;
    
public:
    void add_compressor(std::unique_ptr<ICompressor<T>> compressor) {
        compressors_.push_back(std::move(compressor));
    }
    
    std::vector<uint8_t> compress(const T& data) const override {
        if (compressors_.empty()) {
            // Fallback: raw serialization
            const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(&data);
            return std::vector<uint8_t>(data_ptr, data_ptr + sizeof(T));
        }
        
        // Apply compressors in sequence
        std::vector<uint8_t> result = compressors_[0]->compress(data);
        
        // Note: For proper composite compression, we'd need to handle 
        // type conversions between different compressor stages
        return result;
    }
    
    T decompress(const std::vector<uint8_t>& compressed) const override {
        if (compressors_.empty()) {
            return *reinterpret_cast<const T*>(compressed.data());
        }
        
        return compressors_[0]->decompress(compressed);
    }
    
    size_t compressed_size(const T& data) const override {
        if (compressors_.empty()) return sizeof(T);
        return compressors_[0]->compressed_size(data);
    }
    
    double compression_ratio(const T& data) const override {
        if (compressors_.empty()) return 1.0;
        return compressors_[0]->compression_ratio(data);
    }
};

} // namespace memory
} // namespace replay_buffer