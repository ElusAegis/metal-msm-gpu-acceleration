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

inline SerBn254Point toSerBn254Point(const thread BN254Point& val) {
    return *(const thread SerBn254Point*)(&val);
}

inline BN254Point fromSerBn254Point(const threadgroup SerBn254Point& val) {
    return *(const threadgroup BN254Point*)(&val);
}

inline BN254Point fromSerBn254Point(const thread SerBn254Point& val) {
    return *(const thread BN254Point*)(&val);
}

struct PairAccum {
    uint32_t x;
    SerBn254Point val;
};

struct PartialReductionResult {
    uint32_t count;
    SerBn254Point sum;
    SerBn254Point sos;
};