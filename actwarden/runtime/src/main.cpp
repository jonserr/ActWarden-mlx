// actwarden CLI: stream spans from a JSONL trace, pack features per session
// window, score (if a model is given), decide, and emit one JSON decision
// per span to stdout.
//
//   actwarden --trace fixtures/traces/runaway_loop.jsonl \
//             --spec  feature_spec/v1.json \
//             [--model model.mlxfn] [--emit-features] \
//             [--shadow] [--audit-log audit.jsonl] [--bench N]
//
// Modes:
//   default      decisions are emitted as-is (enforcement is the caller's job)
//   --shadow     the engine still evaluates everything, but the effective
//                action is always "continue"; the would-be action is reported
//                as "proposed_action". This is the recommended first
//                deployment mode: observe before enforcing.
//   --audit-log  append every decision (with its feature snapshot and spec
//                version) to a tamper-evident JSONL hash chain; see audit.h
//                and actwarden.replay.verify_audit_log.
//   --bench      after streaming, time pack(+score) on each session's final
//                window for N iterations; p50/p99 to stderr as JSON.
//
// Without --model the engine runs rules-only: that mode is the permanent
// baseline the learned scorer must beat in replay evaluation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "actwarden/feature_packer.h"
#include "actwarden/policy.h"
#include "actwarden/scorer.h"
#include "actwarden/span.h"
#include "audit.h"

using json = nlohmann::json;
namespace aw = actwarden;

namespace {

struct Options {
  std::string trace_path;
  std::string spec_path;
  std::string model_path; // empty => rules-only
  std::string audit_log_path;
  bool emit_features = false;
  bool shadow = false;
  int bench_iters = 0;
};

int usage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s --trace <spans.jsonl> --spec <v1.json> "
      "[--model <model.mlxfn>] [--emit-features] [--shadow] "
      "[--audit-log <audit.jsonl>] [--bench <iters>]\n",
      argv0);
  return 2;
}

std::optional<Options> parse_args(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> const char* {
      return ++i < argc ? argv[i] : nullptr;
    };
    if (a == "--trace") {
      if (const char* v = value()) {
        opt.trace_path = v;
      } else {
        return std::nullopt;
      }
    } else if (a == "--spec") {
      if (const char* v = value()) {
        opt.spec_path = v;
      } else {
        return std::nullopt;
      }
    } else if (a == "--model") {
      if (const char* v = value()) {
        opt.model_path = v;
      } else {
        return std::nullopt;
      }
    } else if (a == "--audit-log") {
      if (const char* v = value()) {
        opt.audit_log_path = v;
      } else {
        return std::nullopt;
      }
    } else if (a == "--bench") {
      if (const char* v = value()) {
        opt.bench_iters = std::atoi(v);
      } else {
        return std::nullopt;
      }
    } else if (a == "--emit-features") {
      opt.emit_features = true;
    } else if (a == "--shadow") {
      opt.shadow = true;
    } else {
      return std::nullopt;
    }
  }
  if (opt.trace_path.empty() || opt.spec_path.empty()) {
    return std::nullopt;
  }
  return opt;
}

json reasons_to_json(const aw::Decision& decision) {
  json reasons = json::array();
  for (const auto& r : decision.reasons) {
    reasons.push_back(
        {{"code", r.code}, {"detail", r.detail}, {"value", r.value}});
  }
  return reasons;
}

int64_t now_unix_nano() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void run_bench(
    const aw::FeaturePacker& packer,
    const aw::Scorer* scorer,
    const std::unordered_map<std::string, aw::SessionWindow>& sessions,
    int iters) {
  std::vector<double> us;
  us.reserve(static_cast<std::size_t>(iters) * sessions.size());
  for (const auto& [_, window] : sessions) {
    for (int i = 0; i < iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      auto features = packer.pack(window);
      if (scorer != nullptr) {
        (void)scorer->score(features);
      }
      const auto t1 = std::chrono::steady_clock::now();
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  }
  if (us.empty()) {
    return;
  }
  std::sort(us.begin(), us.end());
  const json bench = {
      {"bench",
       {{"what", scorer != nullptr ? "pack+score" : "pack"},
        {"iters_per_session", iters},
        {"sessions", sessions.size()},
        {"p50_us", us[us.size() / 2]},
        {"p99_us",
         us[static_cast<std::size_t>(
             std::max(0.0, std::ceil(0.99 * us.size()) - 1))]}}}};
  std::fprintf(stderr, "%s\n", bench.dump().c_str());
}

} // namespace

int main(int argc, char** argv) {
  auto opt = parse_args(argc, argv);
  if (!opt) {
    return usage(argv[0]);
  }

  try {
    auto spec = aw::FeatureSpec::load(opt->spec_path);
    aw::FeaturePacker packer(spec);
    aw::PolicyEngine policy(aw::PolicyConfig{}, spec);
    std::unique_ptr<aw::Scorer> scorer;
    if (!opt->model_path.empty()) {
      scorer = std::make_unique<aw::Scorer>(opt->model_path);
    }
    std::unique_ptr<aw::AuditLog> audit;
    if (!opt->audit_log_path.empty()) {
      audit = std::make_unique<aw::AuditLog>(opt->audit_log_path);
    }

    std::unordered_map<std::string, aw::SessionWindow> sessions;
    std::unordered_map<std::string, int> span_index;
    int record_index = 0;
    aw::TraceReader reader(opt->trace_path);

    while (auto span = reader.next()) {
      const std::string session_id = span->session_id;
      auto [it, _] = sessions.try_emplace(session_id, spec.window_max_spans);
      it->second.add(std::move(*span));

      const auto features = packer.pack(it->second);
      std::optional<aw::RiskScores> scores;
      if (scorer) {
        scores = scorer->score(features);
      }
      const auto decision =
          policy.decide(it->second, features, scores ? &*scores : nullptr);
      const aw::Action effective =
          opt->shadow ? aw::Action::kContinue : decision.action;

      json rec = {
          {"session_id", session_id},
          {"span_index", span_index[session_id]++},
          {"action", aw::to_string(effective)},
          {"proposed_action", aw::to_string(decision.action)},
          {"shadow", opt->shadow},
          {"reasons", reasons_to_json(decision)},
      };
      if (decision.scored) {
        rec["scores"] = {
            {"loop_risk", decision.scores.loop_risk},
            {"cost_risk", decision.scores.cost_risk},
            {"tool_misuse_risk", decision.scores.tool_misuse_risk},
            {"dead_end_risk", decision.scores.dead_end_risk}};
      }
      if (audit) {
        json audit_rec = rec;
        audit_rec["v"] = 1;
        audit_rec["record_index"] = record_index++;
        audit_rec["ts_unix_nano"] = now_unix_nano();
        audit_rec["spec_version"] = spec.spec_version;
        audit_rec["features"] = features; // full snapshot: replayable
        audit->append(std::move(audit_rec));
      }
      if (opt->emit_features) {
        rec["features"] = features;
      }
      std::cout << rec.dump() << "\n";
    }

    if (opt->bench_iters > 0) {
      run_bench(packer, scorer.get(), sessions, opt->bench_iters);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "actwarden: %s\n", e.what());
    return 1;
  }
  return 0;
}
