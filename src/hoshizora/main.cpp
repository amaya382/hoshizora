#include "hoshizora/app/pagerank.h"
#include "hoshizora/core/compress/multiple.h"
#include "hoshizora/core/executor/bulksync_gas_executor.h"
#include "hoshizora/core/io/io.h"
#include "hoshizora/core/model/graph.h"
#include "hoshizora/core/util/includes.h"
#include <iostream>
#include <utility>

namespace hoshizora {
void main(int argc, char *argv[]) {
  using _Graph = Graph<u32, empty_t, empty_t, f32, f32>;
  init();
  const u32 num_iters =
      argc > 2 ? (u32)std::strtol(argv[2], nullptr, 10) : 1000;
  debug::logger->info("#numa nodes: {}", loop::num_numa_nodes);
  debug::logger->info("#threads: {}", loop::num_threads);
  debug::logger->info("#iters: {}", num_iters);
  debug::point("started");
  auto edge_list = IO::fromFile0(argv[1]);
  debug::point("loaded");
  auto graph = _Graph::from_edge_list(edge_list.data(), edge_list.size());
  debug::point("converted");
  BulkSyncGASExecutor<PageRankKernel<_Graph>> executor(graph, num_iters);
  const auto result = executor.run();
  debug::point("done");
  debug::logger->info(result);

  debug::report("started", "loaded");
  debug::report("loaded", "converted");
  debug::report("converted", "done");
  // loop::quit();

  /*
  compress::a32_vector<u8> compressed(6000000);
  debug::point("encstart");
  compress::multiple::encode(ints.data(), offsets.data(), 3, compressed.data());
  debug::point("encend");
  compress::a32_vector<u32> decompressed((3000001 + 31) / 32 * 32);
  compress::a32_vector<u32> decompressed_offsets(8);
  debug::point("decstart");
  compress::multiple::decode(compressed.data(), 3, decompressed.data(),
                             decompressed_offsets.data());
  debug::point("decend");

  debug::print("encstart", "encend");
  debug::print("decstart", "decend");

  u32 consumed = 0;
  debug::point("enc start");
  for (u32 i = 0; i < 100; ++i) {
    consumed += compress::multiple::encode(ints.data(), offsets.data(), 3,
                                           compressed.data());
  }
  debug::point("enc end");

  u32 consumed0 = 0;
  debug::point("est start");
  for (u32 i = 0; i < 100; ++i) {
    consumed0 += compress::multiple::estimate(ints.data(), offsets.data(), 3);
  }
  debug::point("est end");

  debug::logger->info("ans: {}, est: {}", consumed, consumed0);
  debug::print("enc start", "enc end");
  debug::report("est start", "est end");
   */
}
} // namespace hoshizora

int main(int argc, char *argv[]) {
  hoshizora::main(argc, argv);
  return 0;
}
