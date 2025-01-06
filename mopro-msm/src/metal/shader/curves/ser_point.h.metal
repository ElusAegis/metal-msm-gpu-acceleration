#pragma once

#include "../arithmetics/unsigned_int.h.metal"
#include "../fields/fp_bn254.h.metal"

namespace {
    typedef UnsignedInteger<8> u256;
    typedef UnsignedInteger<4> u128;
    typedef FpBN254 FieldElement;
    typedef ECPoint<FieldElement, 0, u256> BN254Point;

}

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

inline UnsignedInteger<8> fromSerU256(const thread SerU256& s) {
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

inline BN254Point fromSerBn254Point(const thread SerBn254Point& s) {
    BN254Point p;
    p.x = FpBN254(fromSerU256(s.x));
    p.y = FpBN254(fromSerU256(s.y));
    p.z = FpBN254(fromSerU256(s.z));
    return p;
}

//inline UnsignedInteger<8> hide4ValuesIntoUnsignedInteger(uint x, uint y, uint z, uint w) {
//    uint placeholder = 1000000;
//    UnsignedInteger<8> x_val;
//    x_val.m_limbs[0] = x;
//    x_val.m_limbs[1] = placeholder;
//    x_val.m_limbs[2] = y;
//    x_val.m_limbs[3] = placeholder;
//    x_val.m_limbs[4] = z;
//    x_val.m_limbs[5] = placeholder;
//    x_val.m_limbs[6] = w;
//    x_val.m_limbs[7] = placeholder;
//    return x_val;
//}
//
//inline BN254Point hide12ValuesIntoBN254Point(uint x1, uint x2, uint x3, uint x4, uint y1, uint y2, uint y3, uint y4, uint z1, uint z2, uint z3, uint z4) {
//    BN254Point p;
//    p.x = FpBN254(hide4ValuesIntoUnsignedInteger(x1, x2, x3, x4));
//    p.y = FpBN254(hide4ValuesIntoUnsignedInteger(y1, y2, y3, y4));
//    p.z = FpBN254(hide4ValuesIntoUnsignedInteger(z1, z2, z3, z4));
//    return p;
//}

struct PairAccum {
    uint32_t x;
    SerBn254Point val;
};

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