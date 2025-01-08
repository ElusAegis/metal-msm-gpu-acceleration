#ifndef unsigned_int_h
#define unsigned_int_h

#include <metal_stdlib>

template <const uint64_t NUM_LIMBS>
struct UnsignedInteger {
    metal::array<uint32_t, NUM_LIMBS> m_limbs;

    constexpr UnsignedInteger() = default;

    constexpr static UnsignedInteger from_int(uint32_t n) {
        UnsignedInteger res;
        res.m_limbs = {};
        res.m_limbs[NUM_LIMBS - 1] = n;
        return res;
    }

    constexpr static UnsignedInteger from_int(uint64_t n) {
        UnsignedInteger res;
        res.m_limbs = {};
        res.m_limbs[NUM_LIMBS - 2] = (uint32_t)(n >> 32);
        res.m_limbs[NUM_LIMBS - 1] = (uint32_t)(n & 0xFFFFFFFF);
        return res;
    }

    constexpr static UnsignedInteger from_bool(bool b) {
        UnsignedInteger res;
        res.m_limbs = {};
        if (b) {
            res.m_limbs[NUM_LIMBS - 1] = 1;
        }
        return res;
    }

    constexpr static UnsignedInteger from_high_low(UnsignedInteger high, UnsignedInteger low) {
        UnsignedInteger res = low;

        for (uint64_t i = 0; i < NUM_LIMBS / 2; i++) {
            res.m_limbs[i] = high.m_limbs[i + NUM_LIMBS / 2];
        }

        return res;
    }

    constexpr UnsignedInteger low() const {
        UnsignedInteger res = *this;

        for (uint64_t i = 0; i < NUM_LIMBS / 2; i++) {
            res.m_limbs[i] = 0;
        }

        return res;
    }

    constexpr UnsignedInteger high() const {
        UnsignedInteger res;
        res.m_limbs = {};

        for (uint64_t i = 0; i < NUM_LIMBS / 2; i++) {
            res.m_limbs[NUM_LIMBS / 2 + i] = m_limbs[i];
        }

        return res;
    }

    static UnsignedInteger max() {
        UnsignedInteger res = {};

        for (uint64_t i = 0; i < NUM_LIMBS; i++) {
            res.m_limbs[i] = 0xFFFFFFFF;
        }

        return res;
    }

    constexpr UnsignedInteger operator+(const UnsignedInteger rhs) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs {};
        uint64_t carry = 0;
        int i = NUM_LIMBS;

        while (i > 0) {
            uint64_t c = uint64_t(m_limbs[i - 1]) + uint64_t(rhs.m_limbs[i - 1]) + carry;
            limbs[i - 1] = c & 0xFFFFFFFF;
            carry = c >> 32;
            i -= 1;
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr bool operator==(const UnsignedInteger rhs) const
    {
        for (uint32_t i = 0; i < NUM_LIMBS; i++) {
            if (m_limbs[i] != rhs.m_limbs[i]) {
                return false;
            }
        }
        return true;
    }

    constexpr UnsignedInteger operator+=(const UnsignedInteger rhs)
    {
        *this = *this + rhs;
        return *this;
    }

    constexpr UnsignedInteger operator-(const UnsignedInteger rhs) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs {};
        uint64_t carry = 0;
        uint64_t i = NUM_LIMBS;

        while (i > 0) {
            i -= 1;
            int64_t c = (int64_t)(m_limbs[i]) - (int64_t)(rhs.m_limbs[i]) + carry;
            limbs[i] = c & 0xFFFFFFFF;
            carry = c < 0 ? -1 : 0;
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr UnsignedInteger operator-=(const UnsignedInteger rhs)
    {
        *this = *this - rhs;
        return *this;
    }

    constexpr UnsignedInteger operator*(const UnsignedInteger rhs) const
    {
        long int INT_NUM_LIMBS = (long int)NUM_LIMBS;
        uint64_t n = 0;
        uint64_t t = 0;

        for (long int i = INT_NUM_LIMBS - 1; i >= 0; i--) {
            if (m_limbs[i] != 0) {
                n = INT_NUM_LIMBS - 1 - i;
            }
            if (rhs.m_limbs[i] != 0) {
                t = INT_NUM_LIMBS - 1 - i;
            }
        }

        metal::array<uint32_t, NUM_LIMBS> limbs {};

        uint64_t carry = 0;
        for (uint64_t i = 0; i <= t; i++) {
            for (uint64_t j = 0; j <= n; j++) {
                uint64_t uv = (uint64_t)(limbs[NUM_LIMBS - 1 - (i + j)])
                    + (uint64_t)(m_limbs[NUM_LIMBS - 1 - j])
                        * (uint64_t)(rhs.m_limbs[NUM_LIMBS - 1 - i])
                    + carry;
                carry = uv >> 32;
                limbs[NUM_LIMBS - 1 - (i + j)] = uv & 0xFFFFFFFF;
            }
            if (i + n + 1 < NUM_LIMBS) {
                limbs[NUM_LIMBS - 1 - (i + n + 1)] = carry & 0xFFFFFFFF;
                carry = 0;
            }
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    uint64_t cast(uint32_t n) {
      return ((uint64_t)n) >> 32;
    }

    constexpr UnsignedInteger operator*=(const UnsignedInteger rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    constexpr UnsignedInteger operator&(const UnsignedInteger rhs) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs {};
        for (uint32_t i = 0; i < NUM_LIMBS; i++) {
            limbs[i] = m_limbs[i] & rhs.m_limbs[i];
        }
        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr UnsignedInteger operator<<(const uint32_t times) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs {};
        uint32_t a = times / 32;
        uint32_t b = times % 32;

        if (b == 0) {
            int64_t i = 0;
            while (i < (int64_t)NUM_LIMBS - (int64_t)a) {
                limbs[i] = m_limbs[a + i];
                i += 1;
            }
        } else {
            limbs[NUM_LIMBS - 1 - a] = m_limbs[NUM_LIMBS - 1] << b;
            uint64_t i = a + 1;
            while (i < NUM_LIMBS) {
                limbs[NUM_LIMBS - 1 - i] = (m_limbs[NUM_LIMBS - 1 - i + a] << b) | (m_limbs[NUM_LIMBS - i + a] >> (32 - b));
                i += 1;
            }
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr UnsignedInteger operator>>(const uint32_t times) const
    {
        metal::array<uint32_t, NUM_LIMBS> limbs {};
        uint32_t a = times / 32;
        uint32_t b = times % 32;

        if (b == 0) {
            int64_t i = 0;
            while (i < (int64_t)NUM_LIMBS - (int64_t)a) {
                limbs[a + i] = m_limbs[i];
                i += 1;
            }
        } else {
            limbs[a] = m_limbs[0] >> b;
            uint64_t i = a + 1;
            while (i < NUM_LIMBS) {
                limbs[i] = (m_limbs[i - a - 1] << (32 - b)) | (m_limbs[i - a] >> b);
                i += 1;
            }
        }

        return UnsignedInteger<NUM_LIMBS> {limbs};
    }

    constexpr bool operator>(const UnsignedInteger rhs) const {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return true;
        if (m_limbs[i] < rhs.m_limbs[i]) return false;
      }

      return false;
    }

    constexpr bool operator>=(const UnsignedInteger rhs) {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return true;
        if (m_limbs[i] < rhs.m_limbs[i]) return false;
      }

      return true;
    }

    constexpr bool operator<(const UnsignedInteger rhs) const {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return false;
        if (m_limbs[i] < rhs.m_limbs[i]) return true;
      }

      return false;
    }

    constexpr bool operator<=(const UnsignedInteger rhs) const {
      for (uint64_t i = 0; i < NUM_LIMBS; i++) {
        if (m_limbs[i] > rhs.m_limbs[i]) return false;
        if (m_limbs[i] < rhs.m_limbs[i]) return true;
      }

      return true;
    }

    /// Test if bit `bit_index` is set (1).
    /// bit_index=0 is the least significant bit.
    constexpr bool test_bit(uint32_t bit_index) const {
        if (bit_index >= 32 * NUM_LIMBS) {
            return false; // out of range
        }
        // find the limb and the bit offset
        // Example: limb  = (NUM_LIMBS - 1) - (bit_index / 32)
        // Because the LSD is m_limbs[NUM_LIMBS-1].
        uint32_t limb = (NUM_LIMBS - 1u) - (bit_index / 32u);
        uint32_t offset = bit_index % 32u;
        return (m_limbs[limb] >> offset) & 1u;
    }

    /// Extract up to 32 bits starting at `start` (LSB=0).
    /// Returns the extracted bits as a uint32_t.
    /// Example usage for a c-bit window: extract_bits(w*C, C).
    constexpr uint32_t extract_bits(uint32_t start, uint32_t width) const {
        // We'll only implement a 0 < width <= 32 version.
        // Combine up to two limbs if the window crosses a limb boundary.
        if (width == 0) {
            return 0;
        }
        if (start + width > 32 * NUM_LIMBS) {
            // Out of range => return 0 or clamp?
            // We'll clamp here for safety or return 0
            return 0;
        }

        // A simple approach: gather bits one by one
        // (not the fastest but easy to read)
        uint32_t result = 0u;
        for (uint32_t i = 0; i < width; i++) {
            bool bit = test_bit(start + i);
            if (bit) {
                result |= (1u << i);
            }
        }
        return result;
    }
};

#endif /* unsigned_int_h */
