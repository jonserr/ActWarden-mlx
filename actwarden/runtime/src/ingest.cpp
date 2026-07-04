// JSONL span ingestion and feature-spec loading. The only TU (besides
// main.cpp's output path) that touches the JSON library.

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "actwarden/feature_packer.h"
#include "actwarden/span.h"

namespace actwarden {

using json = nlohmann::json;

SpanKind span_kind_from_string(const std::string& s) {
  if (s == "LLM")
    return SpanKind::kLLM;
  if (s == "TOOL")
    return SpanKind::kTool;
  if (s == "RETRIEVER")
    return SpanKind::kRetriever;
  if (s == "AGENT")
    return SpanKind::kAgent;
  if (s == "CHAIN")
    return SpanKind::kChain;
  return SpanKind::kOther;
}

const char* to_string(SpanKind k) {
  switch (k) {
    case SpanKind::kLLM:
      return "LLM";
    case SpanKind::kTool:
      return "TOOL";
    case SpanKind::kRetriever:
      return "RETRIEVER";
    case SpanKind::kAgent:
      return "AGENT";
    case SpanKind::kChain:
      return "CHAIN";
    case SpanKind::kOther:
      return "OTHER";
  }
  return "OTHER";
}

namespace {

Span span_from_json(const json& d) {
  Span s;
  s.trace_id = d.at("trace_id").get<std::string>();
  s.span_id = d.at("span_id").get<std::string>();
  if (d.contains("parent_span_id") && d["parent_span_id"].is_string()) {
    s.parent_span_id = d["parent_span_id"].get<std::string>();
  }
  s.name = d.value("name", "");
  s.start_time_unix_nano = d.at("start_time_unix_nano").get<int64_t>();
  s.end_time_unix_nano = d.at("end_time_unix_nano").get<int64_t>();
  s.error = d.value("status_code", "OK") == "ERROR";
  s.session_id = d.value("session_id", s.trace_id);

  const json attrs = d.value("attributes", json::object());
  std::string kind = d.value("kind", "");
  if (kind.empty())
    kind = attrs.value("openinference.span.kind", "OTHER");
  s.kind = span_kind_from_string(kind);

  s.tool_name = attrs.value("tool.name", "");
  s.model_name = attrs.value("llm.model_name", "");
  s.prompt_tokens = attrs.value("llm.token_count.prompt", int64_t{0});
  s.completion_tokens = attrs.value("llm.token_count.completion", int64_t{0});
  s.cost_usd = attrs.value("actwarden.cost_usd", 0.0);
  if (attrs.contains("retrieval.document_scores")) {
    for (const auto& v : attrs["retrieval.document_scores"]) {
      s.document_scores.push_back(v.get<float>());
    }
  }
  return s;
}

} // namespace

struct TraceReader::Impl {
  std::ifstream in;
  std::string path;
  int lineno = 0;
};

TraceReader::TraceReader(const std::string& jsonl_path)
    : impl_(std::make_unique<Impl>()) {
  impl_->in.open(jsonl_path);
  impl_->path = jsonl_path;
  if (!impl_->in) {
    throw std::runtime_error("TraceReader: cannot open " + jsonl_path);
  }
}

TraceReader::~TraceReader() = default;
TraceReader::TraceReader(TraceReader&&) noexcept = default;
TraceReader& TraceReader::operator=(TraceReader&&) noexcept = default;

std::optional<Span> TraceReader::next() {
  std::string line;
  while (std::getline(impl_->in, line)) {
    ++impl_->lineno;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos)
      continue;
    try {
      return span_from_json(json::parse(line));
    } catch (const std::exception& e) {
      throw std::runtime_error(
          impl_->path + ":" + std::to_string(impl_->lineno) +
          ": malformed span: " + e.what());
    }
  }
  return std::nullopt;
}

std::size_t FeatureSpec::index_of(const std::string& feature_name) const {
  for (std::size_t i = 0; i < feature_names.size(); ++i) {
    if (feature_names[i] == feature_name)
      return i;
  }
  throw std::out_of_range("FeatureSpec: unknown feature " + feature_name);
}

FeatureSpec FeatureSpec::load(const std::string& json_path) {
  std::ifstream in(json_path);
  if (!in)
    throw std::runtime_error("FeatureSpec: cannot open " + json_path);
  json j = json::parse(in);

  FeatureSpec spec;
  spec.spec_version = j.at("spec_version").get<std::string>();
  spec.window_max_spans = j.at("window").at("max_spans").get<std::size_t>();

  const auto& features = j.at("features");
  spec.feature_names.resize(features.size());
  for (const auto& f : features) {
    const auto idx = f.at("index").get<std::size_t>();
    if (idx >= spec.feature_names.size() || !spec.feature_names[idx].empty()) {
      throw std::runtime_error(
          "FeatureSpec: non-contiguous or duplicate index");
    }
    spec.feature_names[idx] = f.at("name").get<std::string>();
  }
  for (const auto& h : j.at("risk_heads")) {
    spec.risk_head_names.push_back(h.at("name").get<std::string>());
  }

  const auto& n = j.at("normalization");
  spec.budget_usd = n.at("budget_usd").get<double>();
  spec.token_rate_per_s = n.at("token_rate_per_s").get<double>();
  spec.latency_ref_ms = n.at("latency_ref_ms").get<double>();
  spec.low_confidence_threshold =
      n.at("low_confidence_threshold").get<double>();
  spec.min_window_seconds = n.at("min_window_seconds").get<double>();
  spec.max_depth = n.at("max_depth").get<int>();
  spec.max_fanout = n.at("max_fanout").get<int>();
  spec.max_error_run = n.at("max_error_run").get<int>();
  spec.loop_ngram = n.at("loop_ngram").get<int>();
  spec.entropy_ref_tools = n.at("entropy_ref_tools").get<int>();
  return spec;
}

} // namespace actwarden
