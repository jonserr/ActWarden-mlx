// Rules-only build of the runtime (CMake: -DACTWARDEN_ENABLE_MLX=OFF).
// Selected instead of scorer.cpp; constructing a Scorer is a hard error so a
// misconfigured deployment cannot silently pretend to have learned scoring.
// The policy engine, packer, CLI, audit log, and parity tests all work in
// this mode — rules-only is the permanent baseline, not a degraded mode.

#include "actwarden/scorer.h"

#include <stdexcept>

namespace actwarden {

struct Scorer::Impl {};

Scorer::Scorer(const std::string& mlxfn_path) {
  throw std::runtime_error(
      "Scorer: this binary was built without MLX (ACTWARDEN_ENABLE_MLX=OFF); "
      "cannot load " +
      mlxfn_path);
}

Scorer::~Scorer() = default;
Scorer::Scorer(Scorer&&) noexcept = default;
Scorer& Scorer::operator=(Scorer&&) noexcept = default;

RiskScores Scorer::score(const std::vector<float>&) const {
  throw std::runtime_error("Scorer: built without MLX");
}

} // namespace actwarden
