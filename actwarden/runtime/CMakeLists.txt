# ActWarden C++ runtime. Builds against the in-tree MLX (this repo is a fork
# of ml-explore/mlx; the fork exists so the runtime, and any future custom
# extensions, develop against MLX source directly).
#
#   macOS (Apple Silicon, Metal):
#     cmake -S actwarden/runtime -B actwarden/runtime/build && cmake --build actwarden/runtime/build -j
#   Linux CPU:
#     cmake -S actwarden/runtime -B actwarden/runtime/build -DMLX_BUILD_METAL=OFF
#   Linux CUDA:
#     cmake -S actwarden/runtime -B actwarden/runtime/build -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=ON
#
# The binary lands at actwarden/runtime/build/actwarden, where the Python
# parity test (python/tests/test_feature_parity.py) expects it.

cmake_minimum_required(VERSION 3.25)
project(actwarden LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Rules-only builds (no MLX, no learned scorer) keep the packer, policy
# engine, CLI, and parity tests buildable anywhere — e.g. minimal CI boxes.
# Loading a --model in such a build is a hard error, never a silent fallback.
option(ACTWARDEN_ENABLE_MLX "Build the MLX-backed scorer" ON)

# --- In-tree MLX -------------------------------------------------------------
if(ACTWARDEN_ENABLE_MLX)
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. ${CMAKE_BINARY_DIR}/mlx)
endif()

# --- JSON (same nlohmann release MLX itself fetches; deduplicated by name) ---
include(FetchContent)
FetchContent_Declare(
  json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# --- Runtime library ----------------------------------------------------------
if(ACTWARDEN_ENABLE_MLX)
  set(ACTWARDEN_SCORER_SRC src/scorer.cpp)
else()
  set(ACTWARDEN_SCORER_SRC src/scorer_disabled.cpp)
endif()

add_library(actwarden_runtime STATIC
  src/ingest.cpp
  src/feature_packer.cpp
  src/policy.cpp
  src/audit.cpp
  ${ACTWARDEN_SCORER_SRC})
target_include_directories(actwarden_runtime
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${json_SOURCE_DIR}/single_include)
if(ACTWARDEN_ENABLE_MLX)
  target_link_libraries(actwarden_runtime PUBLIC mlx)
endif()

# --- CLI ----------------------------------------------------------------------
add_executable(actwarden src/main.cpp)
target_include_directories(actwarden PRIVATE ${json_SOURCE_DIR}/single_include)
target_link_libraries(actwarden PRIVATE actwarden_runtime)

# TODO(slice-day-5): benchmark target (pack+score p50/p99) and C++ unit tests
# via the same googletest MLX already uses.
