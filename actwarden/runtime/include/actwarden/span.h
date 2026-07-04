// ActWarden runtime: span data model and trace ingestion.
//
// Wire format: OpenTelemetry-flavored JSONL with OpenInference semantic
// attributes; see docs/architecture/overview.md ("Data model"). The Python
// mirror is actwarden/python/actwarden/traces.py. An OTLP/HTTP receiver is
// planned to feed the same Span type through the same downstream path.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace actwarden {

enum class SpanKind : uint8_t {
  kLLM,
  kTool,
  kRetriever,
  kAgent,
  kChain,
  kOther
};

SpanKind span_kind_from_string(const std::string& s);
const char* to_string(SpanKind k);

struct Span {
  std::string trace_id;
  std::string span_id;
  std::string parent_span_id; // empty => root
  std::string name;
  SpanKind kind = SpanKind::kOther;
  int64_t start_time_unix_nano = 0;
  int64_t end_time_unix_nano = 0;
  bool error = false; // status_code == "ERROR"
  std::string session_id;

  // Typed semantic attributes the governor consumes.
  std::string tool_name; // "tool.name", empty if absent
  std::string model_name; // "llm.model_name"
  int64_t prompt_tokens = 0; // "llm.token_count.prompt"
  int64_t completion_tokens = 0; // "llm.token_count.completion"
  double cost_usd = 0.0; // "actwarden.cost_usd"
  std::vector<float> document_scores; // "retrieval.document_scores"

  double duration_ms() const {
    return static_cast<double>(end_time_unix_nano - start_time_unix_nano) / 1e6;
  }
};

// Streaming JSONL span reader. Pimpl keeps the JSON dependency out of the
// public interface.
class TraceReader {
 public:
  explicit TraceReader(const std::string& jsonl_path); // throws on open failure
  ~TraceReader();
  TraceReader(TraceReader&&) noexcept;
  TraceReader& operator=(TraceReader&&) noexcept;
  TraceReader(const TraceReader&) = delete;
  TraceReader& operator=(const TraceReader&) = delete;

  // nullopt at EOF. Throws std::runtime_error with file:line on malformed
  // input: a governor must not silently skip evidence.
  std::optional<Span> next();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace actwarden
