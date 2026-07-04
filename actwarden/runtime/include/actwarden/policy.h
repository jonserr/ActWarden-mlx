// ActWarden runtime: policy engine — deterministic rules first, learned
// scores second. The model can only add caution, never remove it. Every
// decision carries structured, stable-ID reasons so control actions are
// auditable and replayable.

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "actwarden/feature_packer.h"
#include "actwarden/scorer.h"
#include "actwarden/span.h"

namespace actwarden {

// Severity-ordered: when several rules/thresholds propose actions, the
// highest value wins. Must match ACTIONS in
// actwarden/python/actwarden/replay.py.
enum class Action : uint8_t {
  kContinue = 0,
  kReduceBudget = 1,
  kSwitchModel = 2,
  kRequireApproval = 3,
  kBlockTool = 4,
  kEscalate = 5,
};

const char* to_string(Action a);

struct Reason {
  std::string code; // stable machine ID, e.g. "R001:budget_exhausted",
                    // "T:loop_risk>=0.80"
  std::string detail; // human-readable explanation
  float value = 0.0f; // the triggering value
};

struct Decision {
  Action action = Action::kContinue;
  std::vector<Reason> reasons; // empty => clean continue
  RiskScores scores; // zeros when running rules-only
  bool scored = false; // false => rules-only decision
};

struct PolicyConfig {
  // Deterministic rules
  double budget_usd = 1.0; // R001: cost_total >= budget => escalate
  int max_error_run = 4; // R004: consecutive errors => require_approval
  int max_depth = 16; // R003: span-tree depth => require_approval
  // R002: adjacent (previous tool -> next tool) pairs that must not occur.
  std::vector<std::pair<std::string, std::string>> tool_pair_denylist = {
      {"web_fetch", "shell_exec"},
  };

  // Learned-score thresholds (calibration + hysteresis: slice day 4/5;
  // without hysteresis a governor flaps, which is worse than being wrong).
  float loop_reduce_budget = 0.60f;
  float loop_require_approval = 0.80f;
  float cost_reduce_budget = 0.70f;
  float misuse_block_tool = 0.80f;
  float dead_end_switch_model = 0.60f;
  float dead_end_escalate = 0.80f;
};

class PolicyEngine {
 public:
  // Feature indices are resolved from the spec once, by name, at construction.
  PolicyEngine(PolicyConfig config, const FeatureSpec& spec);

  // scores == nullptr => rules-only mode (the permanent baseline).
  Decision decide(
      const SessionWindow& window,
      const std::vector<float>& features,
      const RiskScores* scores) const;

 private:
  PolicyConfig config_;
  double spec_max_error_run_;
  std::size_t idx_cost_total_;
  std::size_t idx_error_run_;
  std::size_t idx_depth_max_;
};

} // namespace actwarden
