#include "../arithmetics/unsigned_int.h.metal"
#include "../fields/fp_bn254.h.metal"

namespace {
    typedef UnsignedInteger<8> u256;
    typedef UnsignedInteger<4> u128;
    typedef ECPoint<FieldElement, 0, u256> BN254Point;

}

//struct SerU64 {
//    uint32_t high[2];
//    uint32_t low;
//};
//
//struct SerU128 {
//    SerU64 high;
//    SerU64 low;
//};

struct SerU256 {
    uint32_t u32_limbs[8];
};

struct SerBn254Point {
    SerU256 x;
    SerU256 y;
    SerU256 z;
};

inline SerU256 toSerU256(const thread UnsignedInteger<8>& val) {
    SerU256 out;
    // Copy 8 × 32-bit limbs from val.m_limbs into out.u32_limbs
    for (uint32_t i = 0; i < 8; i++) {
        out.u32_limbs[i] = val.m_limbs[i];
    }
    return out;
}

inline UnsignedInteger<8> fromSerU256(const threadgroup SerU256& s) {
    UnsignedInteger<8> out;
    // Copy 8 × 32-bit limbs from s.u32_limbs into out.m_limbs
    for (uint32_t i = 0; i < 8; i++) {
        out.m_limbs[i] = s.u32_limbs[i];
    }
    return out;
}

inline SerBn254Point toSerBn254Point(const thread BN254Point& p) {
    SerBn254Point pt;
    pt.x = toSerU256(p.x.inner);
    pt.y = toSerU256(p.y.inner);
    pt.z = toSerU256(p.z.inner);
    return pt;
}

inline BN254Point fromSerBn254Point(const threadgroup SerBn254Point& s) {
    BN254Point p;
    p.x = FpBN254(fromSerU256(s.x));
    p.y = FpBN254(fromSerU256(s.y));
    p.z = FpBN254(fromSerU256(s.z));
    return p;
}

//
//inline SerU256 toSerU256(const u256& val) {
//    SerU256 s;
//    s.high.high = val.high.high;
//    s.high.low  = val.high.low;
//    s.low.high  = val.low.high;
//    s.low.low   = val.low.low;
//    return s;
//}
//
//inline u256 fromSerU256(const SerU256& s) {
//    // Reconstruct two u128 parts, then build a u256
//    u128 h(s.high.high, s.high.low);
//    u128 l(s.low.high, s.low.low);
//    return u256(h, l);
//}