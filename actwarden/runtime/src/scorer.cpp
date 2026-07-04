// MLX-backed scorer: imports the exported .mlxfn graph (weights baked in)
// and evaluates it. The only TU in the runtime that includes MLX.
//
// Import pattern follows examples/export/eval_mlp.cpp upstream.

#include "actwarden/scorer.h"

#include <stdexcept>

#include <mlx/mlx.h>

namespace mx = mlx::core;

namespace actwarden {

struct Scorer::Impl {
  mx::ImportedFunction fn;
  explicit Impl(const std::string& path) : fn(mx::import_function(path)) {}
};

Scorer::Scorer(const std::string& mlxfn_path) {
  try {
    impl_ = std::make_unique<Impl>(mlxfn_path);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Scorer: failed to import " + mlxfn_path + ": " + e.what());
  }
}

Scorer::~Scorer() = default;
Scorer::Scorer(Scorer&&) noexcept = default;
Scorer& Scorer::operator=(Scorer&&) noexcept = default;

RiskScores Scorer::score(const std::vector<float>& features) const {
  auto x = mx::array(
      features.begin(), {1, static_cast<int>(features.size())}, mx::float32);
  auto out = impl_->fn({x})[0];
  out.eval();
  if (out.size() != 4) {
    throw std::runtime_error(
        "Scorer: expected 4 risk heads, got " + std::to_string(out.size()));
  }
  const float* p = out.data<float>();
  return RiskScores{p[0], p[1], p[2], p[3]};
}

} // namespace actwarden
