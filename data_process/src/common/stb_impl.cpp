// Single translation unit that compiles the STB implementations once.
// Every other translation unit includes <stb_image.h>/<stb_image_write.h>
// WITHOUT defining either implementation macro.
//
// The library is built with hidden visibility. The focused visibility pragma
// below keeps the STB entry points importable by the unit tests (which link the
// shared library) without routing them through an artificial wrapper. The
// definitions still exist exactly once.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility push(default)
#endif
#include <stb_image.h>
#include <stb_image_write.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC visibility pop
#endif
