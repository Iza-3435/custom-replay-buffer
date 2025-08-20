cmake_minimum_required(VERSION 3.12)
project(ReplayBuffer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=native")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -mfma")
endif()

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/replay_buffer/*.cpp")
file(GLOB_RECURSE HEADERS "include/replay_buffer/*.hpp")

# Create library
add_library(replay_buffer SHARED ${SOURCES} ${HEADERS})
add_library(ReplayBuffer::ReplayBuffer ALIAS replay_buffer)

# Set target properties
set_target_properties(replay_buffer PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${HEADERS}"
)

# Include directories for target
target_include_directories(replay_buffer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Install rules
include(GNUInstallDirs)

install(TARGETS replay_buffer
    EXPORT ReplayBufferTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/replay_buffer
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export targets
install(EXPORT ReplayBufferTargets
    FILE ReplayBufferTargets.cmake
    NAMESPACE ReplayBuffer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ReplayBuffer
)

# Create config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReplayBufferConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ReplayBufferConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ReplayBuffer
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ReplayBufferConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ReplayBufferConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ReplayBufferConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ReplayBuffer
)

# Build examples and tests if this is the main project
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(BUILD_EXAMPLES "Build examples" ON)
    option(BUILD_TESTS "Build tests" ON)
    
    if(BUILD_EXAMPLES)
        add_subdirectory(examples)
    endif()
    
    if(BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()
endif()

# Print build information
message(STATUS "ReplayBuffer Library Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CXX Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")