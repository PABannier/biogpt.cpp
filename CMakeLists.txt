cmake_minimum_required(VERSION 3.12)
project("biogpt.cpp" C CXX)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BIOGPT_STANDALONE ON)
else()
    set(BIOGPT_STANDALONE OFF)
endif()

option(BIOGPT_BUILD_EXAMPLES             "biogpt: build examples" ${BIOGPT_STANDALONE})

# Build libraries

set(BIOGPT_LIB biogpt.cpp)

add_subdirectory(ggml)

add_library(
        ${BIOGPT_LIB}
        biogpt
        biogpt.cpp
        biogpt.h
        bpe.cpp
        bpe.h
        mosestokenizer.cpp
        mosestokenizer.h
)

if (BIOGPT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

target_link_libraries(${BIOGPT_LIB} PUBLIC ggml)
target_include_directories(${BIOGPT_LIB} PUBLIC .)
target_compile_features(${BIOGPT_LIB} PUBLIC cxx_std_11)

# Copy Perl Uniprops assets in `data` subdirectory
file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)