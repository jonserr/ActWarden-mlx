// ActWarden runtime: learned risk scoring via an exported MLX function.
//
// The artifact is a .mlxfn produced by actwarden.model.export_for_runtime
// (weights baked in as graph constants; same mechanism as examples/export/).
// MLX types stay out of this header (pimpl): only scorer.cpp compiles
// against mlx, keeping the rest of the runtime buildable and testable
// without it.

#pragma once

#include <memory>
#include <string>
#include <vector>

namespace actwarden {

// Order must match feature_spec/v1.json risk_heads.
struct RiskScores {
  float loop_risk = 0.0f;
  float cost_risk = 0.0f;
  float tool_misuse_risk = 0.0f;
  float dead_end_risk = 0.0f;
};

class Scorer {
 public:
  // Throws std::runtime_error if the artifact cannot be loaded.
  explicit Scorer(const std::string& mlxfn_path);
  ~Scorer();
  Scorer(Scorer&&) noexcept;
  Scorer& operator=(Scorer&&) noexcept;
  Scorer(const Scorer&) = delete;
  Scorer& operator=(const Scorer&) = delete;

  // features.size() must equal the exported input dim (FeatureSpec::dim()).
  // Latency budget: sub-millisecond on CPU for the v1 MLP — enforced by
  // benchmark, not by promise (bench hook: slice day 5).
  RiskScores score(const std::vector<float>& features) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace actwarden
