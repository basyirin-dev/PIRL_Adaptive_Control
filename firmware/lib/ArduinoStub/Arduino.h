#pragma once
#include <cstdint>
#include <cstdio>
#include <cmath>

struct SerialStub {
    void begin(uint32_t) {}

    template<typename T>
    void print(T) {}

    template<typename T>
    void print(T, int) {}

    template<typename T>
    void println(T) {}
    void println() {}

    template<typename... Args>
    void printf(const char* fmt, Args... args) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"
        ::printf(fmt, args...);
#pragma GCC diagnostic pop
    }

    operator bool() { return true; }
};

// inline variable: one definition across all translation units (C++14 compat
// via __attribute__ or just upgrade flags — see note below)
inline SerialStub Serial;

struct IntervalTimer {
    template<typename F>
    void begin(F, uint32_t) {}
};

inline uint32_t micros() { return 0; }
inline uint32_t millis() { return 0; }

#ifndef OUTPUT_MAX
  #define OUTPUT_MAX  12.0f
#endif
#ifndef OUTPUT_MIN
  #define OUTPUT_MIN -12.0f
#endif
