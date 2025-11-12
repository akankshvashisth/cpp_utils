#pragma once

#include "double4d.hpp"
#include <print>

template <> struct std::formatter<aks::dbl4, char> {
  bool full = false;

  template <class ParseContext>
  constexpr ParseContext::iterator parse(ParseContext &ctx) {
    auto it = ctx.begin();
    if (it == ctx.end())
      return it;

    if (*it == '#') {
      full = true;
      ++it;
    }
    if (it != ctx.end() && *it != '}')
      throw std::format_error("Invalid format args for dbl4.");

    return it;
  }

  template <class FmtContext>
  FmtContext::iterator format(aks::dbl4 s, FmtContext &ctx) const {
    std::string out;
    if (!full) {
      std::stringstream ss;
      ss << s;
      out = ss.str();
    } else
      out = std::format("[{}, {}, {}, {}]", s[0], s[1], s[2], s[3]);

    return std::ranges::copy(std::move(out), ctx.out()).out;
  }
};
