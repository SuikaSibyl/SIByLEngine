#ifndef _SRENDERER_CPP_COMPATIBLE_HEADER_
#define _SRENDERER_CPP_COMPATIBLE_HEADER_

#ifdef __cplusplus
#define ternary(X, Y, Z) ternary(X, Y, Z)
#define enum_macro enum struct
#define in_ref(X) X const &
#define out_ref(X) X &
#define inout_ref(X) X &
#define mutating
#else
#define ternary(X, Y, Z) (select((X), (Y), (Z)))
#define enum_macro enum
#define in_ref(X) in X
#define out_ref(X) out X
#define inout_ref(X) inout X
// #define select(a, b, c) ((a) ? (b) : (c))
#define mutating [mutating]
#endif

#endif // _SRENDERER_CPP_COMPATIBLE_HEADER_