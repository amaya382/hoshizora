#ifndef HOSHIZORA_PRIMITIVE_INCLUDES_H
#define HOSHIZORA_PRIMITIVE_INCLUDES_H

#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#ifdef __linux__
#include "external/pcm/cpucounters.h"
#include <sched.h>
#elif __APPLE__
#include <cpuid.h>
#include <mach/thread_act.h>
#endif
#ifdef SUPPORT_NUMA
#include <numa.h>
#endif
#include "external/spdlog/spdlog.h"

namespace hoshizora {
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;
using empty_t = std::nullptr_t[0];

namespace debug {
static const auto logger = spdlog::stderr_color_mt("hoshizora");

#ifdef __linux__
static const auto pcm = PCM::getInstance();

static inline void init_logger() {
  logger->set_level(spdlog::level::debug);
  pcm->program(PCM::DEFAULT_EVENTS, nullptr);
}
#else
static inline void init_logger() { logger->set_level(spdlog::level::debug); }
#endif

struct partial_score {
  const std::chrono::high_resolution_clock::time_point time;

#ifdef __linux__
  const SystemCounterState counter;

  explicit partial_score(std::chrono::high_resolution_clock::time_point time,
                         SystemCounterState counter)
      : time(time), counter(counter) {}
#else
  explicit partial_score(std::chrono::high_resolution_clock::time_point time)
      : time(time) {}
#endif
};

std::unordered_map<std::string, std::unique_ptr<partial_score>> scores;

void point(const std::string &key) {
  logger->info(key);
  const auto time = std::chrono::high_resolution_clock::now();
#ifdef __linux__
  const auto counter = getSystemCounterState();

  scores[key] = std::make_unique<partial_score>(time, counter);
#else
  scores[key] = std::make_unique<partial_score>(time);
#endif
}

void print(const std::string &start_key, const std::string &end_key) {
#ifdef __linux__
  const auto start = scores[start_key]->counter;
  const auto end = scores[end_key]->counter;
  logger->info("\n"
               "[{} -> {}]\n"
               "ElapsedTime[sec]:\t{}\n"
               "Read[GB]:\t{}\n"
               "Write[GB]:\t{}\n"
               "L2CacheMisses:\t{}\n"
               "L3CacheMisses:\t{}\n"
               "L2CacheHitRatio:\t{}\n"
               "L3CacheHitRatio:\t{}\n"
               "InstructionsRetired:\t{}\n"
               "IPC:\t{}",
               start_key, end_key,
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   scores[end_key]->time - scores[start_key]->time)
                       .count() /
                   1000.0,
               getBytesReadFromMC(start, end) / 1024.0 / 1024.0 / 1024.0,
               getBytesWrittenToMC(start, end) / 1024.0 / 1024.0 / 1024.0,
               getL2CacheMisses(start, end), getL3CacheMisses(start, end),
               getL2CacheHitRatio(start, end), getL3CacheHitRatio(start, end),
               getInstructionsRetired(start, end), getIPC(start, end));
#else
  logger->info("\n"
               "[{} -> {}]\n"
               "ElapsedTime[sec]:\t{}",
               start_key, end_key,
               std::chrono::duration_cast<std::chrono::milliseconds>(
                   scores[end_key]->time - scores[start_key]->time)
                       .count() /
                   1000.0);
#endif
}
} // namespace debug

namespace sched {
static inline i32 get_cpu_id() {
#ifdef __linux__
  return sched_getcpu();
#elif __APPLE__
  std::array<u32, 4> cpu_info = {{0, 0, 0, 0}};
  __cpuid_count(1, 0, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
  i32 cpu_id = -1;
  if ((cpu_info[3] & (1 << 9)) == 0) {
    /* no APIC on chip */
  } else {
    cpu_id = static_cast<u32>(cpu_info[1] >> 24);
  }
  if (cpu_id < 0)
    cpu_id = 0;
  return cpu_id;
#else
  return -1;
#endif
}
} // namespace sched

namespace loop {
static constexpr bool support_numa =
#ifdef SUPPORT_NUMA
    true;
#else
    false;
#endif
static const u32 num_threads =
    std::thread::hardware_concurrency() / 2; // TODO: kill HT
static const u32 num_numa_nodes =
#ifdef SUPPORT_NUMA
    numa_num_configured_nodes();
#else
    1;
#endif
static const std::vector<u32> numa_boundaries = {0, 2, 4};
} // namespace loop

namespace mem {
template <class T> static inline T *malloc(u64 length) {
#ifdef SUPPORT_NUMA
  return static_cast<T *>(numa_alloc_local(sizeof(T) * length));
#else
  return static_cast<T *>(std::malloc(sizeof(T) * length));
#endif
}

template <class T> static inline T *malloc(u64 length, u32 node) {
#ifdef SUPPORT_NUMA
  return static_cast<T *>(numa_alloc_onnode(sizeof(T) * length, node));
#else
  return static_cast<T *>(std::malloc(sizeof(T) * length));
#endif
}

template <class T> static inline T *calloc(u64 length) {
  auto arr = malloc<T>(length);
  std::memset(arr, 0, sizeof(T) * length);
  return arr;
}

template <class T> static inline T *calloc(u64 length, u32 node) {
  auto arr = malloc<T>(length, node);
  std::memset(arr, 0, sizeof(T) * length);
  return arr;
}

static inline void free(void *ptr, size_t size) {
#ifdef SUPPORT_NUMA
  numa_free(ptr, size);
#else
  std::free(ptr);
#endif
}
} // namespace mem

namespace mock {
//[[deprecated("Mock")]]
static inline u32 thread_to_numa(u32 thread_id) {
  return thread_id < 2 ? 0 : 1;
}
} // namespace mock
} // namespace hoshizora

#endif // HOSHIZORA_PRIMITIVE_INCLUDES_H
