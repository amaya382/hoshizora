#ifndef SINGLE_H
#define SINGLE_H

#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string>

#include "hoshizora/core/compress/common.h"
#include "hoshizora/core/util/includes.h"

namespace hoshizora::compress::single {
/*
 * encode
 */
static u32 encode(const u32 *__restrict const in, const u32 size,
                  u8 *__restrict const out) {
  if (size > 0) {
    u32 in_offset = 0;
    u32 out_offset = 0;
    const u32 n_blocks = size / 8u;
    if (n_blocks) {
      const u32 n_flag_blocks = (n_blocks + 1u) / 2u;
      const u32 n_flag_blocks_align32 = ((n_flag_blocks + 31u) / 32u) * 32u;
      out_offset += n_flag_blocks_align32;

      a32_vector<u8> flags(n_blocks + 1u); // TODO: w/o vector

      u8 n_used_bits = 0;
      auto prev = _mm256_setzero_si256();
      auto reg =
          _mm256_load_si256(reinterpret_cast<__m256ipc>(out + out_offset));
      alignas(32) u32 xs[LENGTH];
      for (u32 i = 0; i < n_blocks; i++) {
        const auto curr =
            _mm256_load_si256(reinterpret_cast<__m256icpc>(in + in_offset));
        const auto diff = _mm256_sub_epi32(curr, prev);

        _mm256_store_si256(reinterpret_cast<__m256ipc>(xs),
                           diff); // TODO: reg only
        const u8 pack_idx = pack_sizes_helper[_lzcnt_u32(xs[7])];
        const u32 pack_size = pack_sizes[pack_idx];
        flags[i] = pack_idx;

        if (n_used_bits + pack_size > BIT_PER_BOX) {
          // flush
          _mm256_store_si256(reinterpret_cast<__m256ipc>(out + out_offset),
                             reg);
          out_offset += YMM_BYTE;

          // mv next
          reg = diff;
          n_used_bits = static_cast<u8>(pack_size);
        } else {
          reg = _mm256_or_si256(reg, _mm256_slli_epi32(diff, n_used_bits));
          n_used_bits += pack_size;
        }

        prev = _mm256_permutevar8x32_epi32(curr, broadcast_mask);
        in_offset += LENGTH;
      }

      if (n_used_bits > 0) {
        // flush
        _mm256_store_si256(reinterpret_cast<__m256ipc>(out + out_offset), reg);
        out_offset += YMM_BYTE;
      }

      const u32 N = n_flag_blocks / YMM_BYTE;
      for (u32 i = 0; i < N; i++) {
        const auto acc = _mm256_or_si256(
            _mm256_load_si256(reinterpret_cast<__m256icpc>(
                flags.data() + YMM_BYTE * (i * 2u))),
            _mm256_slli_epi8(_mm256_load_si256(reinterpret_cast<__m256icpc>(
                                 flags.data() + YMM_BYTE * (i * 2u + 1))),
                             4u));
        _mm256_store_si256(reinterpret_cast<__m256ipc>(out + YMM_BYTE * i),
                           acc);
      }
      for (u32 i = N * YMM_BYTE * 2u; i < n_blocks; i += 2u) {
        flags[i] |= flags[i + 1u] << 4u; // cannot manage odd
        out[i / 2u] = flags[i];
      }
    }

    const u32 remain = size - in_offset;
    if (remain > 0) {
      const u32 flag_idx = out_offset;
      out_offset++; // flags
      if (in_offset == 0) {
        if (in[0] <= 0xFFFFu) {
          reinterpret_cast<u16 *__restrict const>(out + out_offset)[0] =
              static_cast<u16>(in[0]);
          out_offset += 2u;
        } else {
          reinterpret_cast<u32 *__restrict const>(out + out_offset)[0] = in[0];
          out_offset += 4u;
          // nibble from lower bit
          out[flag_idx] = 0b00000001u;
        }
        in_offset++;
      }
      for (; in_offset < size; in_offset++) {
        const u32 diff = in[in_offset] - in[in_offset - 1u];
        if (diff <= 0xFFFFu) {
          reinterpret_cast<u16 *__restrict const>(out + out_offset)[0] =
              static_cast<u16>(diff);
          out_offset += 2u;
        } else {
          reinterpret_cast<u32 *__restrict const>(out + out_offset)[0] = diff;
          out_offset += 4u;
          // nibble from lower bit
          out[flag_idx] |= 0b00000001u << (remain - (size - in_offset));
        }
      }

      out_offset += (8u - remain) * 2u; // skip for overrun in decode
      // if not exists, decoder reads out of byte array
    }

    return out_offset;
  } else {
    return 0;
  }
}

static u32 estimate(const u32 *__restrict const in, const u32 size) {
  if (size == 0) {
    return 0;
  }

  u32 in_offset = 0;
  u32 out_offset = 0;
  const u32 n_blocks = size / 8u;

  if (n_blocks) {
    const u32 n_flag_blocks = (n_blocks + 1u) / 2u;
    const u32 n_flag_blocks_align32 = ((n_flag_blocks + 31u) / 32u) * 32u;
    out_offset += n_flag_blocks_align32;

    u8 n_used_bits = 0;
    u32 prev = 0;
    for (u32 i = 0; i < n_blocks; i++) {
      const u32 curr = in[in_offset + 7]; // check only last value

      const u8 pack_idx = pack_sizes_helper[_lzcnt_u32(curr - prev)];
      const u32 pack_size = pack_sizes[pack_idx];

      if (n_used_bits + pack_size > BIT_PER_BOX) {
        out_offset += YMM_BYTE;
        n_used_bits = static_cast<u8>(pack_size);
      } else {
        n_used_bits += pack_size;
      }

      prev = curr;
      in_offset += LENGTH;
    }

    out_offset += (n_used_bits / BIT_PER_BOX) * YMM_BYTE;

    if (n_used_bits > 0) {
      out_offset += YMM_BYTE;
    }
  }

  const u32 remain = size - in_offset;
  if (remain > 0) {
    out_offset++; // flags

    // size < 8
    if (in_offset == 0) {
      if (in[0] <= 0xFFFFu) {
        out_offset += 2u;
      } else {
        out_offset += 4u;
      }
      in_offset++;
    }

    for (; in_offset < size; in_offset++) {
      const u32 diff = in[in_offset] - in[in_offset - 1u];
      if (diff <= 0xFFFFu) {
        out_offset += 2u;
      } else {
        out_offset += 4u;
      }
    }

    out_offset += (8u - remain) * 2u; // skip for overrun in decode
    // if not exists, decoder reads out of byte array
  }

  return out_offset;
}

/*
 * decode
 */
template <u8 f0, u8 f1, u8 f2, u8 f3, u8 f4, u8 f5, u8 f6>
static inline u8 ungap(const u8 *__restrict const in, const u32 prev,
                       u32 *__restrict const out) {
  constexpr u32 o0 = 0;
  constexpr u32 o1 = o0 + (f0 ? 4 : 2);
  constexpr u32 o2 = o1 + (f1 ? 4 : 2);
  constexpr u32 o3 = o2 + (f2 ? 4 : 2);
  constexpr u32 o4 = o3 + (f3 ? 4 : 2);
  constexpr u32 o5 = o4 + (f4 ? 4 : 2);
  constexpr u32 o6 = o5 + (f5 ? 4 : 2);
  constexpr u32 total = o6 + (f6 ? 4 : 2);

  if constexpr (f0) {
    out[0] = prev + reinterpret_cast<const u32 *__restrict const>(in)[0];
  } else {
    out[0] = prev + reinterpret_cast<const u16 *__restrict const>(in)[0];
  }

  if constexpr (f1) {
    out[1] = out[0] + reinterpret_cast<const u32 *__restrict const>(in + o1)[0];
  } else {
    out[1] = out[0] + reinterpret_cast<const u16 *__restrict const>(in + o1)[0];
  }

  if constexpr (f2) {
    out[2] = out[1] + reinterpret_cast<const u32 *__restrict const>(in + o2)[0];
  } else {
    out[2] = out[1] + reinterpret_cast<const u16 *__restrict const>(in + o2)[0];
  }

  if constexpr (f3) {
    out[3] = out[2] + reinterpret_cast<const u32 *__restrict const>(in + o3)[0];
  } else {
    out[3] = out[2] + reinterpret_cast<const u16 *__restrict const>(in + o3)[0];
  }

  if constexpr (f4) {
    out[4] = out[3] + reinterpret_cast<const u32 *__restrict const>(in + o4)[0];
  } else {
    out[4] = out[3] + reinterpret_cast<const u16 *__restrict const>(in + o4)[0];
  }

  if constexpr (f5) {
    out[5] = out[4] + reinterpret_cast<const u32 *__restrict const>(in + o5)[0];
  } else {
    out[5] = out[4] + reinterpret_cast<const u16 *__restrict const>(in + o5)[0];
  }

  if constexpr (f6) {
    out[6] = out[5] + reinterpret_cast<const u32 *__restrict const>(in + o6)[0];
  } else {
    out[6] = out[5] + reinterpret_cast<const u16 *__restrict const>(in + o6)[0];
  }

  return total;
}

const static function<u8(const u8 *__restrict const, const u32,
                         u32 *__restrict const)>
    ungaps[256] = {ungap<0, 0, 0, 0, 0, 0, 0>, ungap<1, 0, 0, 0, 0, 0, 0>,
                   ungap<0, 1, 0, 0, 0, 0, 0>, ungap<1, 1, 0, 0, 0, 0, 0>,
                   ungap<0, 0, 1, 0, 0, 0, 0>, ungap<1, 0, 1, 0, 0, 0, 0>,
                   ungap<0, 1, 1, 0, 0, 0, 0>, ungap<1, 1, 1, 0, 0, 0, 0>,
                   ungap<0, 0, 0, 1, 0, 0, 0>, ungap<1, 0, 0, 1, 0, 0, 0>,
                   ungap<0, 1, 0, 1, 0, 0, 0>, ungap<1, 1, 0, 1, 0, 0, 0>,
                   ungap<0, 0, 1, 1, 0, 0, 0>, ungap<1, 0, 1, 1, 0, 0, 0>,
                   ungap<0, 1, 1, 1, 0, 0, 0>, ungap<1, 1, 1, 1, 0, 0, 0>,
                   ungap<0, 0, 0, 0, 1, 0, 0>, ungap<1, 0, 0, 0, 1, 0, 0>,
                   ungap<0, 1, 0, 0, 1, 0, 0>, ungap<1, 1, 0, 0, 1, 0, 0>,
                   ungap<0, 0, 1, 0, 1, 0, 0>, ungap<1, 0, 1, 0, 1, 0, 0>,
                   ungap<0, 1, 1, 0, 1, 0, 0>, ungap<1, 1, 1, 0, 1, 0, 0>,
                   ungap<0, 0, 0, 1, 1, 0, 0>, ungap<1, 0, 0, 1, 1, 0, 0>,
                   ungap<0, 1, 0, 1, 1, 0, 0>, ungap<1, 1, 0, 1, 1, 0, 0>,
                   ungap<0, 0, 1, 1, 1, 0, 0>, ungap<1, 0, 1, 1, 1, 0, 0>,
                   ungap<0, 1, 1, 1, 1, 0, 0>, ungap<1, 1, 1, 1, 1, 0, 0>,
                   ungap<0, 0, 0, 0, 0, 1, 0>, ungap<1, 0, 0, 0, 0, 1, 0>,
                   ungap<0, 1, 0, 0, 0, 1, 0>, ungap<1, 1, 0, 0, 0, 1, 0>,
                   ungap<0, 0, 1, 0, 0, 1, 0>, ungap<1, 0, 1, 0, 0, 1, 0>,
                   ungap<0, 1, 1, 0, 0, 1, 0>, ungap<1, 1, 1, 0, 0, 1, 0>,
                   ungap<0, 0, 0, 1, 0, 1, 0>, ungap<1, 0, 0, 1, 0, 1, 0>,
                   ungap<0, 1, 0, 1, 0, 1, 0>, ungap<1, 1, 0, 1, 0, 1, 0>,
                   ungap<0, 0, 1, 1, 0, 1, 0>, ungap<1, 0, 1, 1, 0, 1, 0>,
                   ungap<0, 1, 1, 1, 0, 1, 0>, ungap<1, 1, 1, 1, 0, 1, 0>,
                   ungap<0, 0, 0, 0, 1, 1, 0>, ungap<1, 0, 0, 0, 1, 1, 0>,
                   ungap<0, 1, 0, 0, 1, 1, 0>, ungap<1, 1, 0, 0, 1, 1, 0>,
                   ungap<0, 0, 1, 0, 1, 1, 0>, ungap<1, 0, 1, 0, 1, 1, 0>,
                   ungap<0, 1, 1, 0, 1, 1, 0>, ungap<1, 1, 1, 0, 1, 1, 0>,
                   ungap<0, 0, 0, 1, 1, 1, 0>, ungap<1, 0, 0, 1, 1, 1, 0>,
                   ungap<0, 1, 0, 1, 1, 1, 0>, ungap<1, 1, 0, 1, 1, 1, 0>,
                   ungap<0, 0, 1, 1, 1, 1, 0>, ungap<1, 0, 1, 1, 1, 1, 0>,
                   ungap<0, 1, 1, 1, 1, 1, 0>, ungap<1, 1, 1, 1, 1, 1, 0>,
                   ungap<0, 0, 0, 0, 0, 0, 1>, ungap<1, 0, 0, 0, 0, 0, 1>,
                   ungap<0, 1, 0, 0, 0, 0, 1>, ungap<1, 1, 0, 0, 0, 0, 1>,
                   ungap<0, 0, 1, 0, 0, 0, 1>, ungap<1, 0, 1, 0, 0, 0, 1>,
                   ungap<0, 1, 1, 0, 0, 0, 1>, ungap<1, 1, 1, 0, 0, 0, 1>,
                   ungap<0, 0, 0, 1, 0, 0, 1>, ungap<1, 0, 0, 1, 0, 0, 1>,
                   ungap<0, 1, 0, 1, 0, 0, 1>, ungap<1, 1, 0, 1, 0, 0, 1>,
                   ungap<0, 0, 1, 1, 0, 0, 1>, ungap<1, 0, 1, 1, 0, 0, 1>,
                   ungap<0, 1, 1, 1, 0, 0, 1>, ungap<1, 1, 1, 1, 0, 0, 1>,
                   ungap<0, 0, 0, 0, 1, 0, 1>, ungap<1, 0, 0, 0, 1, 0, 1>,
                   ungap<0, 1, 0, 0, 1, 0, 1>, ungap<1, 1, 0, 0, 1, 0, 1>,
                   ungap<0, 0, 1, 0, 1, 0, 1>, ungap<1, 0, 1, 0, 1, 0, 1>,
                   ungap<0, 1, 1, 0, 1, 0, 1>, ungap<1, 1, 1, 0, 1, 0, 1>,
                   ungap<0, 0, 0, 1, 1, 0, 1>, ungap<1, 0, 0, 1, 1, 0, 1>,
                   ungap<0, 1, 0, 1, 1, 0, 1>, ungap<1, 1, 0, 1, 1, 0, 1>,
                   ungap<0, 0, 1, 1, 1, 0, 1>, ungap<1, 0, 1, 1, 1, 0, 1>,
                   ungap<0, 1, 1, 1, 1, 0, 1>, ungap<1, 1, 1, 1, 1, 0, 1>,
                   ungap<0, 0, 0, 0, 0, 1, 1>, ungap<1, 0, 0, 0, 0, 1, 1>,
                   ungap<0, 1, 0, 0, 0, 1, 1>, ungap<1, 1, 0, 0, 0, 1, 1>,
                   ungap<0, 0, 1, 0, 0, 1, 1>, ungap<1, 0, 1, 0, 0, 1, 1>,
                   ungap<0, 1, 1, 0, 0, 1, 1>, ungap<1, 1, 1, 0, 0, 1, 1>,
                   ungap<0, 0, 0, 1, 0, 1, 1>, ungap<1, 0, 0, 1, 0, 1, 1>,
                   ungap<0, 1, 0, 1, 0, 1, 1>, ungap<1, 1, 0, 1, 0, 1, 1>,
                   ungap<0, 0, 1, 1, 0, 1, 1>, ungap<1, 0, 1, 1, 0, 1, 1>,
                   ungap<0, 1, 1, 1, 0, 1, 1>, ungap<1, 1, 1, 1, 0, 1, 1>,
                   ungap<0, 0, 0, 0, 1, 1, 1>, ungap<1, 0, 0, 0, 1, 1, 1>,
                   ungap<0, 1, 0, 0, 1, 1, 1>, ungap<1, 1, 0, 0, 1, 1, 1>,
                   ungap<0, 0, 1, 0, 1, 1, 1>, ungap<1, 0, 1, 0, 1, 1, 1>,
                   ungap<0, 1, 1, 0, 1, 1, 1>, ungap<1, 1, 1, 0, 1, 1, 1>,
                   ungap<0, 0, 0, 1, 1, 1, 1>, ungap<1, 0, 0, 1, 1, 1, 1>,
                   ungap<0, 1, 0, 1, 1, 1, 1>, ungap<1, 1, 0, 1, 1, 1, 1>,
                   ungap<0, 0, 1, 1, 1, 1, 1>, ungap<1, 0, 1, 1, 1, 1, 1>,
                   ungap<0, 1, 1, 1, 1, 1, 1>, ungap<1, 1, 1, 1, 1, 1, 1>,
                   ungap<0, 0, 0, 0, 0, 0, 0>, ungap<1, 0, 0, 0, 0, 0, 0>,
                   ungap<0, 1, 0, 0, 0, 0, 0>, ungap<1, 1, 0, 0, 0, 0, 0>,
                   ungap<0, 0, 1, 0, 0, 0, 0>, ungap<1, 0, 1, 0, 0, 0, 0>,
                   ungap<0, 1, 1, 0, 0, 0, 0>, ungap<1, 1, 1, 0, 0, 0, 0>,
                   ungap<0, 0, 0, 1, 0, 0, 0>, ungap<1, 0, 0, 1, 0, 0, 0>,
                   ungap<0, 1, 0, 1, 0, 0, 0>, ungap<1, 1, 0, 1, 0, 0, 0>,
                   ungap<0, 0, 1, 1, 0, 0, 0>, ungap<1, 0, 1, 1, 0, 0, 0>,
                   ungap<0, 1, 1, 1, 0, 0, 0>, ungap<1, 1, 1, 1, 0, 0, 0>,
                   ungap<0, 0, 0, 0, 1, 0, 0>, ungap<1, 0, 0, 0, 1, 0, 0>,
                   ungap<0, 1, 0, 0, 1, 0, 0>, ungap<1, 1, 0, 0, 1, 0, 0>,
                   ungap<0, 0, 1, 0, 1, 0, 0>, ungap<1, 0, 1, 0, 1, 0, 0>,
                   ungap<0, 1, 1, 0, 1, 0, 0>, ungap<1, 1, 1, 0, 1, 0, 0>,
                   ungap<0, 0, 0, 1, 1, 0, 0>, ungap<1, 0, 0, 1, 1, 0, 0>,
                   ungap<0, 1, 0, 1, 1, 0, 0>, ungap<1, 1, 0, 1, 1, 0, 0>,
                   ungap<0, 0, 1, 1, 1, 0, 0>, ungap<1, 0, 1, 1, 1, 0, 0>,
                   ungap<0, 1, 1, 1, 1, 0, 0>, ungap<1, 1, 1, 1, 1, 0, 0>,
                   ungap<0, 0, 0, 0, 0, 1, 0>, ungap<1, 0, 0, 0, 0, 1, 0>,
                   ungap<0, 1, 0, 0, 0, 1, 0>, ungap<1, 1, 0, 0, 0, 1, 0>,
                   ungap<0, 0, 1, 0, 0, 1, 0>, ungap<1, 0, 1, 0, 0, 1, 0>,
                   ungap<0, 1, 1, 0, 0, 1, 0>, ungap<1, 1, 1, 0, 0, 1, 0>,
                   ungap<0, 0, 0, 1, 0, 1, 0>, ungap<1, 0, 0, 1, 0, 1, 0>,
                   ungap<0, 1, 0, 1, 0, 1, 0>, ungap<1, 1, 0, 1, 0, 1, 0>,
                   ungap<0, 0, 1, 1, 0, 1, 0>, ungap<1, 0, 1, 1, 0, 1, 0>,
                   ungap<0, 1, 1, 1, 0, 1, 0>, ungap<1, 1, 1, 1, 0, 1, 0>,
                   ungap<0, 0, 0, 0, 1, 1, 0>, ungap<1, 0, 0, 0, 1, 1, 0>,
                   ungap<0, 1, 0, 0, 1, 1, 0>, ungap<1, 1, 0, 0, 1, 1, 0>,
                   ungap<0, 0, 1, 0, 1, 1, 0>, ungap<1, 0, 1, 0, 1, 1, 0>,
                   ungap<0, 1, 1, 0, 1, 1, 0>, ungap<1, 1, 1, 0, 1, 1, 0>,
                   ungap<0, 0, 0, 1, 1, 1, 0>, ungap<1, 0, 0, 1, 1, 1, 0>,
                   ungap<0, 1, 0, 1, 1, 1, 0>, ungap<1, 1, 0, 1, 1, 1, 0>,
                   ungap<0, 0, 1, 1, 1, 1, 0>, ungap<1, 0, 1, 1, 1, 1, 0>,
                   ungap<0, 1, 1, 1, 1, 1, 0>, ungap<1, 1, 1, 1, 1, 1, 0>,
                   ungap<0, 0, 0, 0, 0, 0, 1>, ungap<1, 0, 0, 0, 0, 0, 1>,
                   ungap<0, 1, 0, 0, 0, 0, 1>, ungap<1, 1, 0, 0, 0, 0, 1>,
                   ungap<0, 0, 1, 0, 0, 0, 1>, ungap<1, 0, 1, 0, 0, 0, 1>,
                   ungap<0, 1, 1, 0, 0, 0, 1>, ungap<1, 1, 1, 0, 0, 0, 1>,
                   ungap<0, 0, 0, 1, 0, 0, 1>, ungap<1, 0, 0, 1, 0, 0, 1>,
                   ungap<0, 1, 0, 1, 0, 0, 1>, ungap<1, 1, 0, 1, 0, 0, 1>,
                   ungap<0, 0, 1, 1, 0, 0, 1>, ungap<1, 0, 1, 1, 0, 0, 1>,
                   ungap<0, 1, 1, 1, 0, 0, 1>, ungap<1, 1, 1, 1, 0, 0, 1>,
                   ungap<0, 0, 0, 0, 1, 0, 1>, ungap<1, 0, 0, 0, 1, 0, 1>,
                   ungap<0, 1, 0, 0, 1, 0, 1>, ungap<1, 1, 0, 0, 1, 0, 1>,
                   ungap<0, 0, 1, 0, 1, 0, 1>, ungap<1, 0, 1, 0, 1, 0, 1>,
                   ungap<0, 1, 1, 0, 1, 0, 1>, ungap<1, 1, 1, 0, 1, 0, 1>,
                   ungap<0, 0, 0, 1, 1, 0, 1>, ungap<1, 0, 0, 1, 1, 0, 1>,
                   ungap<0, 1, 0, 1, 1, 0, 1>, ungap<1, 1, 0, 1, 1, 0, 1>,
                   ungap<0, 0, 1, 1, 1, 0, 1>, ungap<1, 0, 1, 1, 1, 0, 1>,
                   ungap<0, 1, 1, 1, 1, 0, 1>, ungap<1, 1, 1, 1, 1, 0, 1>,
                   ungap<0, 0, 0, 0, 0, 1, 1>, ungap<1, 0, 0, 0, 0, 1, 1>,
                   ungap<0, 1, 0, 0, 0, 1, 1>, ungap<1, 1, 0, 0, 0, 1, 1>,
                   ungap<0, 0, 1, 0, 0, 1, 1>, ungap<1, 0, 1, 0, 0, 1, 1>,
                   ungap<0, 1, 1, 0, 0, 1, 1>, ungap<1, 1, 1, 0, 0, 1, 1>,
                   ungap<0, 0, 0, 1, 0, 1, 1>, ungap<1, 0, 0, 1, 0, 1, 1>,
                   ungap<0, 1, 0, 1, 0, 1, 1>, ungap<1, 1, 0, 1, 0, 1, 1>,
                   ungap<0, 0, 1, 1, 0, 1, 1>, ungap<1, 0, 1, 1, 0, 1, 1>,
                   ungap<0, 1, 1, 1, 0, 1, 1>, ungap<1, 1, 1, 1, 0, 1, 1>,
                   ungap<0, 0, 0, 0, 1, 1, 1>, ungap<1, 0, 0, 0, 1, 1, 1>,
                   ungap<0, 1, 0, 0, 1, 1, 1>, ungap<1, 1, 0, 0, 1, 1, 1>,
                   ungap<0, 0, 1, 0, 1, 1, 1>, ungap<1, 0, 1, 0, 1, 1, 1>,
                   ungap<0, 1, 1, 0, 1, 1, 1>, ungap<1, 1, 1, 0, 1, 1, 1>,
                   ungap<0, 0, 0, 1, 1, 1, 1>, ungap<1, 0, 0, 1, 1, 1, 1>,
                   ungap<0, 1, 0, 1, 1, 1, 1>, ungap<1, 1, 0, 1, 1, 1, 1>,
                   ungap<0, 0, 1, 1, 1, 1, 1>, ungap<1, 0, 1, 1, 1, 1, 1>,
                   ungap<0, 1, 1, 1, 1, 1, 1>, ungap<1, 1, 1, 1, 1, 1, 1>};

static u32 decode(const u8 *__restrict const in, const u32 size,
                  u32 *__restrict const out) {
  if (size > 0u) {
    u32 out_offset = 0;
    u32 in_offset = 0;
    const u32 n_blocks = size / LENGTH;

    if (n_blocks) {
      const u32 n_flag_blocks = (n_blocks + 1u) / 2u;
      const u32 n_flag_blocks_align32 = ((n_flag_blocks + 31u) / 32u) * 32u;
      in_offset += n_flag_blocks_align32;

      a32_vector<u8> flags(n_blocks + 1u);
      const u32 N = n_flag_blocks / YMM_BYTE;
      for (u32 i = 0; i < N; i++) {
        const auto reg =
            _mm256_load_si256(reinterpret_cast<__m256icpc>(in + YMM_BYTE * i));
        _mm256_store_si256(
            reinterpret_cast<__m256ipc>(flags.data() + YMM_BYTE * (i * 2u)),
            _mm256_and_si256(
                reg,
                _mm256_load_si256(reinterpret_cast<__m256icpc>(mask8r[4]))));
        _mm256_store_si256(reinterpret_cast<__m256ipc>(
                               flags.data() + YMM_BYTE * (i * 2u + 1u)),
                           _mm256_srli_epi8(reg, 4u));
      }
      for (u32 i = N * YMM_BYTE * 2u; i < n_blocks; i += 2u) {
        flags[i] = static_cast<u8>(in[i / 2u] & 0xFu);
        flags[i + 1u] = in[i / 2u] >> 4u;
      }

      u8 n_used_bits = 0u;
      auto prev = _mm256_setzero_si256();
      auto reg =
          _mm256_load_si256(reinterpret_cast<__m256icpc>(in + in_offset));
      for (u32 i = 0; i < n_blocks; i++) {
        const u32 pack_size = pack_sizes[flags[i]];

        if (n_used_bits + pack_size > BIT_PER_BOX) {
          // mv next
          in_offset += YMM_BYTE;
          reg = _mm256_load_si256(reinterpret_cast<__m256icpc>(in + in_offset));
          const auto for_store = _mm256_add_epi32(
              prev, _mm256_and_si256(
                        reg, _mm256_load_si256(reinterpret_cast<__m256icpc>(
                                 mask32r[pack_size]))));
          _mm256_stream_si256(reinterpret_cast<__m256ipc>(out + out_offset),
                              for_store);
          prev = _mm256_permutevar8x32_epi32(for_store, broadcast_mask);
          n_used_bits = static_cast<u8>(pack_size);
        } else {
          // continue using curr
          const auto for_store = _mm256_add_epi32(
              prev,
              _mm256_and_si256(_mm256_srli_epi32(reg, n_used_bits),
                               _mm256_load_si256(reinterpret_cast<__m256icpc>(
                                   mask32r[pack_size]))));
          _mm256_stream_si256(reinterpret_cast<__m256ipc>(out + out_offset),
                              for_store);
          prev = _mm256_permutevar8x32_epi32(for_store, broadcast_mask);
          n_used_bits += pack_size;
        }

        out_offset += LENGTH;
      }

      if (n_used_bits > 0u) {
        in_offset += YMM_BYTE;
      }
    }

    const u32 remain = size - out_offset;
    if (remain > 0u) {
      const u32 prev = out_offset == 0 ? 0 : out[out_offset - 1u];
      const u32 consumed =
          ungaps[in[in_offset]](in + in_offset + 1u, prev, out + out_offset);

      in_offset += 1u + consumed;
      out_offset += LENGTH - 1u;

      // revert overrun
      in_offset -= (out_offset - size) * 2u;
      /*out_offset -= out_offset - size;*/
    }

    return in_offset;
  } else {
    return 0;
  }
}

template <typename Func>
static u32 traverse(const u8 *__restrict const in, const u32 size, Func f) {
  if (size > 0u) {
    u32 out_offset = 0;
    u32 in_offset = 0;
    const u32 n_blocks = size / LENGTH;

    alignas(32) u32 out[LENGTH];

    if (n_blocks) {
      const u32 n_flag_blocks = (n_blocks + 1u) / 2u;
      const u32 n_flag_blocks_align32 = ((n_flag_blocks + 31u) / 32u) * 32u;
      in_offset += n_flag_blocks_align32;

      a32_vector<u8> flags(n_blocks + 1u);
      const u32 N = n_flag_blocks / YMM_BYTE;
      for (u32 i = 0; i < N; i++) {
        const auto reg =
            _mm256_load_si256(reinterpret_cast<__m256icpc>(in + YMM_BYTE * i));
        _mm256_store_si256(
            reinterpret_cast<__m256ipc>(flags.data() + YMM_BYTE * (i * 2u)),
            _mm256_and_si256(
                reg,
                _mm256_load_si256(reinterpret_cast<__m256icpc>(mask8r[4]))));
        _mm256_store_si256(reinterpret_cast<__m256ipc>(
                               flags.data() + YMM_BYTE * (i * 2u + 1u)),
                           _mm256_srli_epi8(reg, 4u));
      }
      for (u32 i = N * YMM_BYTE * 2u; i < n_blocks; i += 2u) {
        flags[i] = static_cast<u8>(in[i / 2u] & 0xFu);
        flags[i + 1u] = in[i / 2u] >> 4u;
      }

      u8 n_used_bits = 0u;
      auto prev = _mm256_setzero_si256();
      auto reg =
          _mm256_load_si256(reinterpret_cast<__m256icpc>(in + in_offset));
      for (auto i = 0ul; i < n_blocks; i++) {
        const u32 pack_size = pack_sizes[flags[i]];

        if (n_used_bits + pack_size > BIT_PER_BOX) {
          // mv next
          in_offset += YMM_BYTE;
          reg = _mm256_load_si256(reinterpret_cast<__m256icpc>(in + in_offset));
          const auto for_store = _mm256_add_epi32(
              prev, _mm256_and_si256(
                        reg, _mm256_load_si256(reinterpret_cast<__m256icpc>(
                                 mask32r[pack_size]))));
          _mm256_store_si256(reinterpret_cast<__m256ipc>(out), for_store);
          for (u32 j = 0; j < LENGTH; j++) {
            f(out[j], out_offset + j);
          }
          prev = _mm256_permutevar8x32_epi32(for_store, broadcast_mask);
          n_used_bits = static_cast<u8>(pack_size);
        } else {
          // continue using curr
          const auto for_store = _mm256_add_epi32(
              prev,
              _mm256_and_si256(_mm256_srli_epi32(reg, n_used_bits),
                               _mm256_load_si256(reinterpret_cast<__m256icpc>(
                                   mask32r[pack_size]))));
          _mm256_store_si256(reinterpret_cast<__m256ipc>(out), for_store);
          for (u32 j = 0; j < LENGTH; j++) {
            f(out[j], out_offset + j);
          }
          prev = _mm256_permutevar8x32_epi32(for_store, broadcast_mask);
          n_used_bits += pack_size;
        }

        out_offset += LENGTH;
      }

      if (n_used_bits > 0u) {
        in_offset += YMM_BYTE;
      }
    }

    const u32 remain = size - out_offset;
    if (remain > 0u) {
      const u32 consumed =
          ungaps[in[in_offset]](in + in_offset + 1u, out[LENGTH - 1u], out);
      for (u32 i = 0; i < remain; i++) {
        f(out[i], out_offset + i);
      }

      in_offset += 1u + consumed;
      out_offset += LENGTH - 1u;

      // revert overrun
      in_offset -= (out_offset - size) * 2u;
      /*out_offset -= out_offset - size;*/
    }

    return in_offset;
  } else {
    return 0;
  }
}
} // namespace hoshizora::compress::single
#endif // SINGLE_H
