// Append-only, tamper-evident audit log (CLI-internal for now; deliberately
// not part of the public include/ surface until the record schema settles).
//
// Each JSONL record carries "hash_prev" = FNV-1a 64 (hex) of the raw bytes of
// the previous line (excluding the trailing newline). The first record chains
// from fnv1a64("") == "cbf29ce484222325". Any byte-level edit to a line breaks
// verification of the line after it; the final line is only protected once a
// subsequent record lands (known limitation of an unanchored hash chain).
//
// Verification / replay counterpart: actwarden.replay.verify_audit_log.

#pragma once

#include <cstdint>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

namespace actwarden {

// FNV-1a 64-bit, lowercase 16-hex-digit output. Must match
// actwarden.replay.fnv1a64 (known vectors pinned in the Python tests).
std::string fnv1a64_hex(const std::string& data);

class AuditLog {
 public:
  // Opens in append mode. If the file already has records, the chain resumes
  // from the last line (O(file) scan on open; acceptable for the slice).
  explicit AuditLog(const std::string& path);

  // Adds "hash_prev", writes one line, flushes. Records must not contain a
  // "hash_prev" key of their own.
  void append(nlohmann::json record);

 private:
  std::ofstream out_;
  std::string prev_hash_;
};

} // namespace actwarden
