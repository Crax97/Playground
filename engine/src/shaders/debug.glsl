#extension GL_EXT_debug_printf: enable

#ifdef ENABLE_DEBUG_PRINT
#define printf debugPrintfEXT
#else
#define printf
#endif