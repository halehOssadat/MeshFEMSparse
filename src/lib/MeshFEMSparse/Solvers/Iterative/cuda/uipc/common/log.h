// MeshFEM replacement for libuipc's uipc/common/log.h (no spdlog/fmt).
#pragma once
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace uipc
{
namespace detail
{
    // Replace each "{}" in `fmt` with the next argument (fmt-style, "{}" only).
    inline void format_to(std::ostringstream& os, std::string_view fmt)
    {
        os << fmt;
    }

    template <typename T, typename... Args>
    void format_to(std::ostringstream& os, std::string_view fmt, T&& value, Args&&... args)
    {
        auto pos = fmt.find("{}");
        if(pos == std::string_view::npos)
        {
            os << fmt;
            return;
        }
        os << fmt.substr(0, pos) << value;
        format_to(os, fmt.substr(pos + 2), std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::string format(std::string_view fmt, Args&&... args)
    {
        std::ostringstream os;
        format_to(os, fmt, std::forward<Args>(args)...);
        return os.str();
    }
}  // namespace detail

namespace logger
{
    template <typename... Args>
    void info(std::string_view fmt, Args&&... args)
    {
        std::cout << "[info] " << detail::format(fmt, std::forward<Args>(args)...) << std::endl;
    }

    template <typename... Args>
    void warn(std::string_view fmt, Args&&... args)
    {
        std::cout << "[warn] " << detail::format(fmt, std::forward<Args>(args)...) << std::endl;
    }
}  // namespace logger
}  // namespace uipc

// libuipc logs and aborts; here we throw so Python gets an exception instead of a dead process.
#define UIPC_ASSERT(condition, ...)                                                  \
    do                                                                               \
    {                                                                                \
        if(!(condition))                                                             \
            throw std::runtime_error(::uipc::detail::format(                         \
                "Assertion " #condition " failed. {} {}({})",                        \
                ::uipc::detail::format(__VA_ARGS__), __FILE__, __LINE__));           \
    } while(0);
