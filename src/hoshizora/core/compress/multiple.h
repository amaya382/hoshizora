#ifndef MULTIPLE_H
#define MULTIPLE_H

#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string>

#include "hoshizora/core/compress/common.h"
#include "hoshizora/core/compress/single.h"
#include "hoshizora/core/util/includes.h"

namespace hoshizora::compress::multiple {
/*
 * encode
 */
// |offsets| should be n_lists + 1
static u32 encode(const u32 *__restrict const in,
                  const u32 *__restrict const offsets, const u32 n_lists,
                  u8 *__restrict const out) {
  u32 out_consumed = 0;
  out_consumed += single::encode(offsets, n_lists + 1u, out);
  out_consumed = ((out_consumed + 31u) / 32u) * 32u;

  u32 i = 0;
  while (true) {
    auto head = in + offsets[i];
    u32 len = offsets[i + 1] - offsets[i];
    if (len > THRESHOLD) {
      out_consumed += single::encode(head, len, out + out_consumed);
      i++;
    } else {
      u32 len_acc = 0;
      while (true) {
        if ((len >= THRESHOLD && len_acc >= 256u) || i == n_lists) {
          break;
        }
        len = offsets[i + 1] - offsets[i];
        len_acc += len;
        i++;
      }
      {
        size_t consumed = len_acc;
        pfor->encodeArray(
            head, len_acc,
            reinterpret_cast<u32 *__restrict const>(out + out_consumed),
            consumed);
        out_consumed += consumed * 4u;
      }
    }

    if (i == n_lists) {
      break;
    } else {
      out_consumed = ((out_consumed + 31u) / 32u) * 32u;
    }
  }
  return out_consumed;
}

// |offsets| should be n_lists + 1
static u32 estimate(const u32 *__restrict const in,
                        const u32 *__restrict const offsets, const u32 n_lists) {
  u32 out_consumed = 0;
  out_consumed += single::estimate(offsets, n_lists + 1u);
  out_consumed = ((out_consumed + 31u) / 32u) * 32u;

  u32 i = 0;
  while (true) {
    auto head = in + offsets[i];
    u32 len = offsets[i + 1] - offsets[i];
    if (len > THRESHOLD) {
      out_consumed += single::estimate(head, len);
      i++;
    } else {
      u32 len_acc = 0;
      while (true) {
        if ((len >= THRESHOLD && len_acc >= 256u) || i == n_lists) {
          break;
        }
        len = offsets[i + 1] - offsets[i];
        len_acc += len;
        i++;
      }
      {
        size_t consumed = len_acc;
        pfor->estimate(head, len_acc); // FIXME
        out_consumed += consumed * 4u;
      }
    }

    if (i == n_lists) {
      break;
    } else {
      out_consumed = ((out_consumed + 31u) / 32u) * 32u;
    }
  }
  return out_consumed;
}

/*
 * decode
 */
alignas(32) u32 buffer[1000000]; // TODO: must be unused

static void decode(const u8 *__restrict in, const u32 n_lists,
                   u32 *__restrict const out, u32 *__restrict const offsets) {
  u32 in_consumed = 0;
  in_consumed += single::decode(in, n_lists + 1u, offsets);
  in_consumed = ((in_consumed + 31u) / 32u) * 32u;

  u32 out_consumed = 0;
  u32 i = 0;
  while (true) {
    auto head = in + in_consumed;
    u32 len = offsets[i + 1] - offsets[i];
    if (len > THRESHOLD) {
      in_consumed += single::decode(head, len, buffer);
      copy(buffer, buffer + len, out + out_consumed);
      out_consumed += len;
      i++;
    } else {
      u32 len_acc = 0;
      while (true) {
        if ((len >= THRESHOLD && len_acc >= 256u) || i == n_lists) {
          break;
        }
        len = offsets[i + 1] - offsets[i];
        len_acc += len;
        i++;
      }
      {
        size_t consumed = len_acc;
        const auto proceeded =
            reinterpret_cast<const u8 *__restrict>(pfor->decodeArray(
                reinterpret_cast<const u32 *__restrict const>(head),
                consumed /*dummy*/, buffer, consumed /*as len_acc*/));
        copy(buffer, buffer + len_acc, out + out_consumed);
        in_consumed += proceeded - head;
        out_consumed += len_acc;
      }
    }

    if (i == n_lists) {
      break;
    } else {
      in_consumed = ((in_consumed + 31u) / 32u) * 32u;
    }
  }
}

template <typename Func>
static void traverse(const u8 *__restrict const in, const u32 n_lists, Func f,
                     u32 *__restrict const out = nullptr,
                     u32 *__restrict const offsets = nullptr) {
  u32 in_consumed = 0;
  in_consumed += single::decode(in, n_lists + 1u, offsets);
  in_consumed = ((in_consumed + 31u) / 32u) * 32u;

  u32 i = 0;
  while (true) {
    auto head = in + in_consumed;
    u32 len = offsets[i + 1] - offsets[i];
    if (len > THRESHOLD) {
      in_consumed += single::traverse(
          head, len, bind(f, placeholders::_1, i, placeholders::_2));
      i++;
    } else {
      u32 len_acc = 0;
      while (true) {
        if ((len >= THRESHOLD && len_acc >= 256u) || i == n_lists) {
          break;
        }
        len = offsets[i + 1] - offsets[i];
        len_acc += len;
        i++;
      }
      {
        size_t consumed = len_acc;
        const auto proceeded = reinterpret_cast<const u8 *__restrict>(
            pfor->mapArray(reinterpret_cast<const u32 *__restrict const>(head),
                           consumed /*dummy*/, buffer, consumed /*as len_acc*/,
                           bind(f, placeholders::_1, i, placeholders::_2)));
        in_consumed += proceeded - head;
      }
    }

    if (i == n_lists) {
      break;
    } else {
      in_consumed = ((in_consumed + 31u) / 32u) * 32u;
    }
  }
}
} // namespace hoshizora::compress::multiple
#endif // MULTIPLE_H
