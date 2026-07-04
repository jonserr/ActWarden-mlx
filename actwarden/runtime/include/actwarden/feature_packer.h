// ActWarden runtime: session windows and feature packing.
//
// FeaturePacker must reproduce the Python reference implementation
// (actwarden/python/actwarden/features.py) exactly, as specified by
// actwarden/feature_spec/v1.json. Parity gate:
// actwarden/python/tests/test_feature_parity.py::test_cpp_parity.

#pragma once

#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "actwarden/span.h"

namespace actwarden {

// Immutable, parsed view of feature_spec/v1.json. Loaded once at startup.
struct FeatureSpec {
  std::string spec_version;
  std::size_t window_max_spans = 64;
  std::vector<std::string> feature_names; // position == feature index
  std::vector<std::string> risk_head_names; // position == head index

  // normalization constants (names mirror the spec)
  double budget_usd = 1.0;
  double token_rate_per_s = 500.0;
  double latency_ref_ms = 30000.0;
  double low_confidence_threshold = 0.3;
  double min_window_seconds = 0.001;
  int max_depth = 16;
  int max_fanout = 8;
  int max_error_run = 8;
  int loop_ngram = 3;
  int entropy_ref_tools = 8;

  std::size_t dim() const {
    return feature_names.size();
  }
  // Index of a feature by spec name; throws std::out_of_range if absent.
  std::size_t index_of(const std::string& feature_name) const;

  static FeatureSpec load(const std::string& json_path); // impl: ingest.cpp
};

// Ring buffer of the most recent spans for one session, in arrival order.
// The caller feeds spans in canonical order (start_time, span_id).
class SessionWindow {
 public:
  explicit SessionWindow(std::size_t max_spans) : max_spans_(max_spans) {}

  void add(Span span) {
    spans_.push_back(std::move(span));
    if (spans_.size() > max_spans_)
      spans_.pop_front();
  }
  const std::deque<Span>& spans() const {
    return spans_;
  }

 private:
  std::size_t max_spans_;
  std::deque<Span> spans_;
};

class FeaturePacker {
 public:
  explicit FeaturePacker(FeatureSpec spec) : spec_(std::move(spec)) {}

  // Packs the window into a spec_.dim() float vector, spec order, all values
  // clamped to [0,1]. Allocation-light: this sits in the tool-call hot path.
  std::vector<float> pack(const SessionWindow& window) const;

  const FeatureSpec& spec() const {
    return spec_;
  }

 private:
  FeatureSpec spec_;
};

} // namespace actwarden
