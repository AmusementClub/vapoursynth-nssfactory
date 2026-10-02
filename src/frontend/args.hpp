// SPDX-License-Identifier: GPL-2.0-only
#pragma once
// Backend-neutral argument helpers. Every backend plugin reports errors as
// "<namespace>.<Filter>: <message>" (namespace "nss" for the CPU plugin), so
// the shared parsers take the namespace instead of hard-coding it.
#include <stdexcept>
#include <string>

namespace nss::frontend {

[[noreturn]] inline void fail(const char* ns, const char* filter, const char* message) {
    throw std::invalid_argument(std::string(ns) + "." + filter + ": " + message);
}

}  // namespace nss::frontend
