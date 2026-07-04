// Policy engine: deterministic rules R001-R004, then learned-score
// thresholds. Highest-severity proposal wins. Python keeps a development
// mirror in actwarden/python/actwarden/replay.py; this file is authoritative.

#include "actwarden/policy.h"

#include <cstdio>

namespace actwarden {

const char* to_string(Action a) {
  switch (a) {
    case Action::kContinue:
      return "continue";
    case Action::kReduceBudget:
      return "reduce_budget";
    case Action::kSwitchModel:
      return "switch_model";
    case Action::kRequireApproval:
      return "require_approval";
    case Action::kBlockTool:
      return "block_tool";
    case Action::kEscalate:
      return "escalate";
  }
  return "continue";
}

namespace {

void propose(Decision& d, Action a, Reason reason) {
  d.reasons.push_back(std::move(reason));
  if (static_cast<uint8_t>(a) > static_cast<uint8_t>(d.action))
    d.action = a;
}

std::string format_threshold_code(const char* head, float threshold) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "T:%s>=%.2f", head, threshold);
  return buf;
}

} // namespace

PolicyEngine::PolicyEngine(PolicyConfig config, const FeatureSpec& spec)
    : config_(std::move(config)),
      spec_max_error_run_(spec.max_error_run),
      idx_cost_total_(spec.index_of("cost_total_norm")),
      idx_error_run_(spec.index_of("error_run_norm")),
      idx_depth_max_(spec.index_of("depth_max_norm")) {}

Decision PolicyEngine::decide(
    const SessionWindow& window,
    const std::vector<float>& features,
    const RiskScores* scores) const {
  Decision d;

  // --- Deterministic rules (always evaluated; every hit is recorded) ---
  if (features[idx_cost_total_] >= 1.0f) {
    propose(
        d,
        Action::kEscalate,
        {"R001:budget_exhausted",
         "cumulative cost reached the budget",
         features[idx_cost_total_]});
  }
  const Span* prev_tool = nullptr;
  for (const auto& s : window.spans()) {
    if (s.kind != SpanKind::kTool)
      continue;
    if (prev_tool != nullptr) {
      for (const auto& [from, to] : config_.tool_pair_denylist) {
        if (prev_tool->tool_name == from && s.tool_name == to) {
          propose(
              d,
              Action::kBlockTool,
              {"R002:tool_pair_denylist",
               "denylisted tool sequence " + from + " -> " + to,
               1.0f});
        }
      }
    }
    prev_tool = &s;
  }
  if (features[idx_depth_max_] >= 1.0f) {
    propose(
        d,
        Action::kRequireApproval,
        {"R003:depth_limit",
         "span tree depth reached the limit",
         features[idx_depth_max_]});
  }
  const float error_run_threshold =
      static_cast<float>(config_.max_error_run / spec_max_error_run_);
  if (features[idx_error_run_] >= error_run_threshold) {
    propose(
        d,
        Action::kRequireApproval,
        {"R004:error_streak",
         "consecutive tool/model failures",
         features[idx_error_run_]});
  }

  // --- Learned scores (can only add caution, never remove it) ---
  if (scores != nullptr) {
    d.scored = true;
    d.scores = *scores;
    const auto& c = config_;
    if (scores->loop_risk >= c.loop_require_approval) {
      propose(
          d,
          Action::kRequireApproval,
          {format_threshold_code("loop_risk", c.loop_require_approval),
           "high probability of a runaway loop",
           scores->loop_risk});
    } else if (scores->loop_risk >= c.loop_reduce_budget) {
      propose(
          d,
          Action::kReduceBudget,
          {format_threshold_code("loop_risk", c.loop_reduce_budget),
           "elevated loop risk",
           scores->loop_risk});
    }
    if (scores->cost_risk >= c.cost_reduce_budget) {
      propose(
          d,
          Action::kReduceBudget,
          {format_threshold_code("cost_risk", c.cost_reduce_budget),
           "cost growth on track to exceed budget",
           scores->cost_risk});
    }
    if (scores->tool_misuse_risk >= c.misuse_block_tool) {
      propose(
          d,
          Action::kBlockTool,
          {format_threshold_code("tool_misuse_risk", c.misuse_block_tool),
           "anomalous or unsafe tool usage pattern",
           scores->tool_misuse_risk});
    }
    if (scores->dead_end_risk >= c.dead_end_escalate) {
      propose(
          d,
          Action::kEscalate,
          {format_threshold_code("dead_end_risk", c.dead_end_escalate),
           "agent appears stuck; human review needed",
           scores->dead_end_risk});
    } else if (scores->dead_end_risk >= c.dead_end_switch_model) {
      propose(
          d,
          Action::kSwitchModel,
          {format_threshold_code("dead_end_risk", c.dead_end_switch_model),
           "low progress; a different model may help",
           scores->dead_end_risk});
    }
    // TODO(slice-day-4): hysteresis — an action must persist for K windows
    // before downgrading, to prevent decision flapping.
  }

  return d;
}

} // namespace actwarden
