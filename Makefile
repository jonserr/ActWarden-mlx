# ActWarden developer entry points. Wraps the CMake/pytest/docker workflows;
# it does not replace the CMake build (MLX requires CMake). Run `make doctor`
# first — it tells you what's installed and how to get what's missing.
#
# Everything is overridable, e.g.:
#   make build JOBS=4
#   make build-rules-only JSON_INC=/opt/homebrew/include
#   make test BUILD_DIR=/somewhere/else

AW        := actwarden
BUILD_DIR ?= $(AW)/runtime/build
PY        ?= python3
CMAKE     ?= cmake
CLI       := $(BUILD_DIR)/actwarden
SPEC      := $(AW)/feature_spec/v1.json
FIXTURE   := $(AW)/fixtures/traces/runaway_loop.jsonl
JOBS      ?= $(shell (nproc || sysctl -n hw.ncpu) 2>/dev/null || echo 4)

UNAME_S := $(shell uname -s)
CMAKE_FLAGS ?=
ifeq ($(UNAME_S),Linux)
  CMAKE_FLAGS += -DMLX_BUILD_METAL=OFF
endif
ifeq ($(CUDA),1)
  CMAKE_FLAGS += -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=ON
endif

# nlohmann/json for the cmake-free rules-only build (the CMake build fetches
# it itself). brew install nlohmann-json / apt-get install nlohmann-json3-dev.
JSON_HPP := $(firstword $(wildcard \
  /opt/homebrew/include/nlohmann/json.hpp \
  /usr/local/include/nlohmann/json.hpp \
  /usr/include/nlohmann/json.hpp))
JSON_INC ?= $(patsubst %/nlohmann/json.hpp,%,$(JSON_HPP))

RUNTIME_SRC := $(AW)/runtime/src
CXX_FILES   := $(shell find $(AW)/runtime -name '*.cpp' -o -name '*.h')
PYTEST      := PYTHONPATH=$(AW)/python $(PY) -m pytest $(AW)/python/tests -q

.DEFAULT_GOAL := help
.PHONY: help doctor build build-rules-only test smoke demo bench \
        format format-check verify docker-cpu docker-cuda clean

help: ## Show this help
	@grep -E '^[a-z][a-zA-Z_-]*:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'

doctor: ## Check prerequisites and print install hints for anything missing
	@command -v $(CMAKE) >/dev/null 2>&1 \
	  && echo "  ok       cmake ($$($(CMAKE) --version | head -1))" \
	  || echo "  MISSING  cmake        -> brew install cmake   (Linux: apt-get install cmake)"
	@command -v $(CXX) >/dev/null 2>&1 \
	  && echo "  ok       $(CXX) ($$($(CXX) --version | head -1))" \
	  || echo "  MISSING  C++ compiler -> xcode-select --install   (Linux: apt-get install build-essential)"
	@command -v $(PY) >/dev/null 2>&1 \
	  && echo "  ok       $(PY) ($$($(PY) --version))" \
	  || echo "  MISSING  python3"
	@$(PY) -c "import pytest" >/dev/null 2>&1 \
	  && echo "  ok       pytest" \
	  || echo "  MISSING  pytest       -> pip install -e '$(AW)/python[dev]'"
	@$(PY) -c "import mlx.core" >/dev/null 2>&1 \
	  && echo "  ok       mlx (python, for training/export)" \
	  || echo "  missing  mlx (python) -> pip install -e '$(AW)/python[train]'  (only needed to train/export)"
	@command -v clang-format >/dev/null 2>&1 \
	  && echo "  ok       clang-format ($$(clang-format --version | head -1))" \
	  || echo "  missing  clang-format -> brew install clang-format  (or: pip install clang-format)"
	@test -n "$(JSON_INC)" \
	  && echo "  ok       nlohmann/json at $(JSON_INC) (for make build-rules-only)" \
	  || echo "  missing  nlohmann/json -> brew install nlohmann-json  (only needed for make build-rules-only)"
	@command -v docker >/dev/null 2>&1 \
	  && echo "  ok       docker" \
	  || echo "  missing  docker       (only needed for make docker-cpu / docker-cuda)"

build: ## Build runtime + in-tree MLX via CMake (Metal on macOS; CUDA=1 for CUDA)
	@command -v $(CMAKE) >/dev/null 2>&1 || { \
	  echo "error: cmake not found. macOS: brew install cmake   Linux: apt-get install cmake"; \
	  echo "       (no cmake yet? 'make build-rules-only' builds the rules-only CLI with just a C++ compiler)"; \
	  exit 1; }
	$(CMAKE) -S $(AW)/runtime -B $(BUILD_DIR) $(CMAKE_FLAGS)
	$(CMAKE) --build $(BUILD_DIR) -j $(JOBS)

build-rules-only: ## Build the rules-only CLI directly with $(CXX) — no CMake, no MLX
	@test -n "$(JSON_INC)" || { \
	  echo "error: nlohmann/json.hpp not found."; \
	  echo "  macOS: brew install nlohmann-json    Linux: apt-get install nlohmann-json3-dev"; \
	  echo "  or pass JSON_INC=/path/containing/nlohmann/"; \
	  exit 1; }
	mkdir -p $(BUILD_DIR)
	$(CXX) -O2 -std=c++20 -I$(AW)/runtime/include -I$(JSON_INC) \
	  $(RUNTIME_SRC)/ingest.cpp $(RUNTIME_SRC)/feature_packer.cpp \
	  $(RUNTIME_SRC)/policy.cpp $(RUNTIME_SRC)/audit.cpp \
	  $(RUNTIME_SRC)/scorer_disabled.cpp $(RUNTIME_SRC)/main.cpp \
	  -o $(CLI)
	@echo "built $(CLI) (rules-only: --model is a hard error in this binary)"

test: ## Run the Python suite; C++ parity/audit tests activate when the CLI exists
	@if [ -x "$(CLI)" ]; then \
	  echo "using C++ CLI: $(CLI)"; \
	  ACTWARDEN_CLI=$(CLI) $(PYTEST); \
	else \
	  echo "note: C++ CLI not built (make build | build-rules-only) — parity tests will skip"; \
	  $(PYTEST); \
	fi

smoke: ## Score the fixture trace end to end (requires a built CLI)
	@test -x "$(CLI)" || { echo "error: $(CLI) not built (make build | build-rules-only)"; exit 1; }
	$(CLI) --trace $(FIXTURE) --spec $(SPEC) | tail -3

demo: ## Instrumented toy agent -> shadow scoring -> audit log -> offline verify
	@test -x "$(CLI)" || { echo "error: $(CLI) not built (make build | build-rules-only)"; exit 1; }
	PYTHONPATH=$(AW)/python $(PY) $(AW)/examples/toy_agent.py /tmp/actwarden_demo_trace.jsonl
	$(CLI) --trace /tmp/actwarden_demo_trace.jsonl --spec $(SPEC) \
	  --shadow --audit-log /tmp/actwarden_demo_audit.jsonl | tail -2
	PYTHONPATH=$(AW)/python $(PY) -c "from actwarden.replay import verify_audit_log; \
v = verify_audit_log('/tmp/actwarden_demo_audit.jsonl', '$(SPEC)'); \
print('audit verify:', 'OK' if v.ok else v.problems, '| records:', v.records)"

bench: ## Latency microbench of the pack(+score) hot path on the fixture
	@test -x "$(CLI)" || { echo "error: $(CLI) not built (make build | build-rules-only)"; exit 1; }
	$(CLI) --trace $(FIXTURE) --spec $(SPEC) --bench 2000 > /dev/null

format: ## Apply the fork's .clang-format to ActWarden C++ sources
	clang-format -i -style=file $(CXX_FILES)

format-check: ## Fail if any ActWarden C++ file deviates from .clang-format
	clang-format -n --Werror -style=file $(CXX_FILES)

verify: format-check test ## format-check + full test suite (pre-commit gate)

docker-cpu: ## Build the Linux CPU image (image build runs the parity suite as a gate)
	docker build -f $(AW)/docker/Dockerfile.linux-cpu -t actwarden:cpu .

docker-cuda: ## Build the Linux CUDA image (best-effort; see Dockerfile notes)
	docker build -f $(AW)/docker/Dockerfile.linux-cuda -t actwarden:cuda .

clean: ## Remove the ActWarden build tree (leaves MLX sources untouched)
	rm -rf $(BUILD_DIR)
