// C++ feature packer. Contract: actwarden/feature_spec/v1.json; reference:
// actwarden/python/actwarden/features.py.
//
// Full parity with the Python reference packer, including canonical window
// ordering by (start_time_unix_nano, span_id). Acceptance gate:
// python/tests/test_feature_parity.py::test_cpp_parity.

#include "actwarden/feature_packer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace actwarden {

namespace {

inline float clamp01(double x) {
  return static_cast<float>(x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x));
}

double entropy_bits(const std::unordered_map<std::string, int>& counts) {
  int total = 0;
  for (const auto& [_, c] : counts) {
    total += c;
  }
  if (total == 0) {
    return 0.0;
  }
  double h = 0.0;
  for (const auto& [_, c] : counts) {
    if (c > 0) {
      const double p = static_cast<double>(c) / total;
      h -= p * std::log2(p);
    }
  }
  return h;
}

// Element at index ceil(0.95*n)-1 of an ascending sort. No interpolation.
double p95(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto idx = static_cast<std::size_t>(
      std::max(0.0, std::ceil(0.95 * static_cast<double>(values.size())) - 1));
  return values[idx];
}

} // namespace

std::vector<float> FeaturePacker::pack(const SessionWindow& window) const {
  std::vector<float> out(spec_.dim(), 0.0f);
  const auto& raw = window.spans();
  const std::size_t n = raw.size();
  if (n == 0) {
    return out;
  }

  // Canonical window order: (start_time_unix_nano, span_id), matching the
  // reference packer regardless of arrival order.
  std::vector<const Span*> spans;
  spans.reserve(n);
  for (const auto& s : raw) {
    spans.push_back(&s);
  }
  std::sort(spans.begin(), spans.end(), [](const Span* a, const Span* b) {
    if (a->start_time_unix_nano != b->start_time_unix_nano) {
      return a->start_time_unix_nano < b->start_time_unix_nano;
    }
    return a->span_id < b->span_id;
  });

  std::size_t n_tool = 0;
  std::size_t n_llm = 0;
  std::size_t n_retriever = 0;
  std::size_t n_error = 0;
  int64_t total_tokens = 0;
  double total_cost = 0.0;
  int64_t window_start = spans.front()->start_time_unix_nano;
  int64_t window_end = spans.front()->end_time_unix_nano;
  std::size_t error_run = 0;
  std::size_t longest_error_run = 0;
  std::vector<std::string> tool_seq;
  std::vector<double> durations_ms;
  durations_ms.reserve(n);

  for (const Span* s : spans) {
    n_tool += s->kind == SpanKind::kTool;
    n_llm += s->kind == SpanKind::kLLM;
    n_retriever += s->kind == SpanKind::kRetriever;
    n_error += s->error;
    total_tokens += s->prompt_tokens + s->completion_tokens;
    total_cost += s->cost_usd;
    window_start = std::min(window_start, s->start_time_unix_nano);
    window_end = std::max(window_end, s->end_time_unix_nano);
    error_run = s->error ? error_run + 1 : 0;
    longest_error_run = std::max(longest_error_run, error_run);
    if (s->kind == SpanKind::kTool) {
      tool_seq.push_back(s->tool_name);
    }
    durations_ms.push_back(s->duration_ms());
  }
  const double window_seconds = std::max(
      static_cast<double>(window_end - window_start) / 1e9,
      spec_.min_window_seconds);

  // 0-4: counts and rates
  out[0] = clamp01(static_cast<double>(n) / spec_.window_max_spans);
  out[1] = clamp01(static_cast<double>(n_tool) / n);
  out[2] = clamp01(static_cast<double>(n_llm) / n);
  out[3] = clamp01(static_cast<double>(n_error) / n);
  out[4] =
      clamp01(static_cast<double>(longest_error_run) / spec_.max_error_run);

  // 5: tool_ngram_repeat_ratio
  const auto ng = static_cast<std::size_t>(spec_.loop_ngram);
  if (tool_seq.size() >= ng) {
    std::set<std::vector<std::string>> unique;
    const std::size_t total = tool_seq.size() - ng + 1;
    for (std::size_t i = 0; i < total; ++i) {
      unique.insert(
          std::vector<std::string>(
              tool_seq.begin() + i, tool_seq.begin() + i + ng));
    }
    out[5] = clamp01(1.0 - static_cast<double>(unique.size()) / total);
  }

  // 6: tool_entropy_norm
  if (!tool_seq.empty()) {
    std::unordered_map<std::string, int> counts;
    for (const auto& name : tool_seq) {
      ++counts[name];
    }
    out[6] = clamp01(entropy_bits(counts) / std::log2(spec_.entropy_ref_tools));
  }

  // 7-9: token velocity, cost total, cost slope
  out[7] = clamp01(
      static_cast<double>(total_tokens) / window_seconds /
      spec_.token_rate_per_s);
  out[8] = clamp01(total_cost / spec_.budget_usd);
  if (n < 2) {
    out[9] = 0.5f;
  } else {
    const std::size_t half = n / 2;
    double first = 0.0;
    double second = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      (i < half ? first : second) += spans[i]->cost_usd;
    }
    out[9] = clamp01(((second - first) / spec_.budget_usd + 1.0) / 2.0);
  }

  // 10: latency_p95_norm
  out[10] = clamp01(p95(std::move(durations_ms)) / spec_.latency_ref_ms);

  // 11-12: retrieval
  out[11] = clamp01(static_cast<double>(n_retriever) / n);
  if (n_retriever > 0) {
    std::size_t low = 0;
    for (const Span* s : spans) {
      if (s->kind != SpanKind::kRetriever) {
        continue;
      }
      const bool low_conf = s->document_scores.empty() ||
          *std::max_element(s->document_scores.begin(),
                            s->document_scores.end()) <
              spec_.low_confidence_threshold;
      low += low_conf;
    }
    out[12] = clamp01(static_cast<double>(low) / n_retriever);
  }

  // 13: depth_max_norm (parent chains resolved within the window; last-wins
  // on duplicate span ids, matching the reference dict build)
  std::unordered_map<std::string, const Span*> by_id;
  for (const Span* s : spans) {
    by_id[s->span_id] = s;
  }
  int max_depth = 1;
  for (const Span* s : spans) {
    int d = 1;
    std::unordered_set<std::string> seen{s->span_id};
    std::string parent = s->parent_span_id;
    while (!parent.empty() && by_id.count(parent) && !seen.count(parent)) {
      ++d;
      seen.insert(parent);
      parent = by_id[parent]->parent_span_id;
    }
    max_depth = std::max(max_depth, d);
  }
  out[13] = clamp01(static_cast<double>(max_depth) / spec_.max_depth);

  // 14: fanout_max_norm
  std::unordered_map<std::string, int> children;
  int max_fanout = 0;
  for (const Span* s : spans) {
    if (!s->parent_span_id.empty() && by_id.count(s->parent_span_id)) {
      max_fanout = std::max(max_fanout, ++children[s->parent_span_id]);
    }
  }
  out[14] = clamp01(static_cast<double>(max_fanout) / spec_.max_fanout);

  // 15: stagnation_norm
  const int64_t duration = window_end - window_start;
  if (duration > 0) {
    std::unordered_set<std::string> seen_keys;
    int64_t last_novel_start = window_start;
    for (const Span* s : spans) {
      const std::string key = std::string(to_string(s->kind)) + '\x1f' +
          (s->kind == SpanKind::kTool ? s->tool_name : s->name);
      if (seen_keys.insert(key).second) {
        last_novel_start = s->start_time_unix_nano;
      }
    }
    out[15] = clamp01(
        static_cast<double>(window_end - last_novel_start) /
        static_cast<double>(duration));
  }

  return out;
}

} // namespace actwarden
