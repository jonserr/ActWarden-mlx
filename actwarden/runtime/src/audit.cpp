#include "audit.h"

#include <cstdio>
#include <stdexcept>

namespace actwarden {

std::string fnv1a64_hex(const std::string& data) {
  uint64_t h = 14695981039346656037ULL;
  for (const unsigned char c : data) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  char buf[17];
  std::snprintf(
      buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return buf;
}

AuditLog::AuditLog(const std::string& path) : prev_hash_(fnv1a64_hex("")) {
  // Resume the chain from the last existing line, if any.
  {
    std::ifstream in(path);
    std::string line;
    std::string last;
    while (std::getline(in, line)) {
      if (!line.empty()) {
        last = line;
      }
    }
    if (!last.empty()) {
      prev_hash_ = fnv1a64_hex(last);
    }
  }
  out_.open(path, std::ios::app);
  if (!out_) {
    throw std::runtime_error("AuditLog: cannot open " + path);
  }
}

void AuditLog::append(nlohmann::json record) {
  record["hash_prev"] = prev_hash_;
  const std::string line = record.dump();
  out_ << line << "\n";
  out_.flush();
  prev_hash_ = fnv1a64_hex(line);
}

} // namespace actwarden
