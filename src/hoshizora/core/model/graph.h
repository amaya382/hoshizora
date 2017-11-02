#ifndef HOSHIZORA_GRAPH_H
#define HOSHIZORA_GRAPH_H

#include "hoshizora/core/util/includes.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

namespace hoshizora {
template <class ID, class VProp, class EProp, class VData, class EData,
          bool IsDirected = true>
struct Graph {
  using _ID = ID;
  using _VProp = VProp;
  using _EProp = EProp;
  using _VData = VData;
  using _EData = EData;
  using _Graph = Graph<ID, VProp, EProp, VData, EData, IsDirected>;

  // TODO
  const u32 num_threads = loop::num_threads;
  const u32 num_numa_nodes = loop::num_numa_nodes;

  ID num_vertices;
  ID num_edges;

  ID *tmp_out_offsets;
  ID *tmp_out_indices;
  ID *tmp_in_offsets;
  ID *tmp_in_indices;

  colle::DiscreteArray<ID> out_degrees;     // [num_vertices]
  colle::DiscreteArray<ID> out_offsets;     // [num_vertices]
  colle::DiscreteArray<ID> out_indices;     // [num_edges]
  colle::DiscreteArray<ID *> out_neighbors; // [num_vertices][degrees[i]]
  colle::DiscreteArray<ID> in_degrees;
  colle::DiscreteArray<ID> in_offsets;
  colle::DiscreteArray<ID> in_indices;
  colle::DiscreteArray<ID *> in_neighbors;

  // colle::DiscreteArray<u8> out_degrees_c;     // [num_vertices]
  colle::DiscreteArray<u8> out_offsets_c; // [num_vertices]
  colle::DiscreteArray<u8> out_indices_c; // [num_edges]
  // colle::DiscreteArray<u8> in_degrees_c;
  colle::DiscreteArray<u8> in_offsets_c;
  colle::DiscreteArray<u8> in_indices_c;

  ID *forward_indices; // [num_edges]
  ID *out_boundaries;
  ID *in_boundaries;
  VProp *v_props;                     // [num_vertices]
  EProp *e_props;                     // [num_edges]
  colle::DiscreteArray<VData> v_data; // [num_vertices]
  colle::DiscreteArray<EData> e_data; // [num_edges]

  colle::DiscreteArray<bool> active_flags; // [num_vertices]

  bool out_degrees_is_initialized = false;
  bool out_offsets_is_initialized = false;
  bool out_indices_is_initialized = false;
  bool out_neighbors_is_initialized = false;
  bool in_degrees_is_initialized = false;
  bool in_offsets_is_initialized = false;
  bool in_indices_is_initialized = false;
  bool in_neighbors_is_initialized = false;
  bool out_boundaries_is_initialized = false;
  bool in_boundaries_is_initialized = false;
  bool forward_indices_is_initialized = false;

  Graph()
      : out_degrees(colle::DiscreteArray<ID>()),
        out_neighbors(colle::DiscreteArray<ID *>()),
        in_degrees(colle::DiscreteArray<ID>()),
        in_neighbors(colle::DiscreteArray<ID *>()),
        v_data(colle::DiscreteArray<VData>()),
        e_data(colle::DiscreteArray<EData>()) {}

  virtual ~Graph() {
    // debug::print("dest graph");
  }

  void set_out_boundaries() {
    assert(!out_offsets_is_initialized);

    const auto chunk_size = num_edges / num_threads;
    out_boundaries = mem::calloc<ID>(num_threads + 1);
    for (u32 thread_id = 1; thread_id < num_threads; ++thread_id) {
      out_boundaries[thread_id] = static_cast<ID>(std::distance(
          tmp_out_offsets,
          std::lower_bound(tmp_out_offsets, tmp_out_offsets + num_vertices,
                           chunk_size * thread_id)));
    }
    out_boundaries[num_threads] = num_vertices;

    out_boundaries_is_initialized = true;
  }

  void set_in_boundaries() {
    assert(!in_offsets_is_initialized);

    const auto chunk_size = num_edges / num_threads;
    in_boundaries = mem::calloc<ID>(num_threads + 1);
    for (u32 thread_id = 1; thread_id < num_threads; ++thread_id) {
      in_boundaries[thread_id] = static_cast<ID>(std::distance(
          tmp_in_offsets,
          std::lower_bound(tmp_in_offsets, tmp_in_offsets + num_vertices,
                           chunk_size * thread_id)));
    }
    in_boundaries[num_threads] = num_vertices;

    in_boundaries_is_initialized = true;
  }

  void set_out_offsets() {
    assert(out_boundaries_is_initialized);

    // FIXME
    if (num_numa_nodes == 1) {
      out_offsets.add(tmp_out_offsets, num_vertices + 1);
    } else {
      loop::each_numa_node(out_boundaries, [&](u32 block_id, u32 numa_id,
                                               u32 lower, u32 upper) {
        // TODO: hide this branch
        const auto size = out_boundaries[num_threads] != upper // not last block
                              ? upper - lower
                              : upper - lower + 1; // w/ cap
        const auto offsets = mem::malloc<u32>(size, numa_id);
        std::memcpy(offsets, tmp_out_offsets + lower, size * sizeof(u32));
        out_offsets.add(offsets, size);
      });

      mem::free(tmp_out_offsets, sizeof(ID) * num_vertices);
    }
    out_offsets_is_initialized = true;
  }

  void set_out_offsets_c() {
    assert(out_boundaries_is_initialized);

    // FIXME
    if (num_numa_nodes == 1) {
      // compress::single_encode(tmp_out_offsets, num_vertices + 1)
      // out_offsets_c.add(, num_vertices + 1);
    } else {
      loop::each_numa_node(out_boundaries, [&](u32 block_id, u32 numa_id,
                                               u32 lower, u32 upper) {
        // TODO: hide this branch
        const auto size = out_boundaries[num_threads] != upper // not last block
                              ? upper - lower
                              : upper - lower + 1; // w/ cap
        const auto offsets = mem::malloc<u32>(size, numa_id);
        std::memcpy(offsets, tmp_out_offsets + lower, size * sizeof(u32));
        out_offsets.add(offsets, size);
      });

      mem::free(tmp_out_offsets, sizeof(ID) * num_vertices);
    }
    out_offsets_is_initialized = true;
  }

  void set_in_offsets() {
    assert(in_boundaries_is_initialized);

    // FIXME
    if (num_numa_nodes == 1) {
      in_offsets.add(tmp_in_offsets, num_vertices + 1);
    } else {
      loop::each_numa_node(in_boundaries, [&](u32 block_id, u32 numa_id,
                                              u32 lower, u32 upper) {
        const auto size = in_boundaries[num_threads] != upper // not last block
                              ? upper - lower
                              : upper - lower + 1; // /w cap
        const auto offsets = mem::malloc<u32>(size, numa_id);
        std::memcpy(offsets, tmp_in_offsets + lower, size * sizeof(u32));
        in_offsets.add(offsets, size);
      });

      mem::free(tmp_in_offsets, sizeof(ID) * num_vertices);
    }
    in_offsets_is_initialized = true;
  }

  void set_out_degrees() {
    assert(out_boundaries_is_initialized);
    assert(out_offsets_is_initialized);
    loop::each_numa_node(
        out_boundaries, [&](u32 block_id, u32 numa_id, u32 lower, u32 upper) {
          const auto size = upper - lower;
          const auto degrees = mem::malloc<ID>(size, numa_id);
          for (u32 i = lower; i < upper - 1; ++i) {
            degrees[i - lower] =
                out_offsets(i + 1, block_id) - out_offsets(i, block_id);
          }
          degrees[size - 1] =
              (out_boundaries[num_threads] != upper // not last block
                   ? out_offsets(upper, block_id + 1)
                   : out_offsets(upper, block_id)) // cap is in curr block
              - out_offsets(upper - 1, block_id);

          out_degrees.add(degrees, size);
        });

    out_degrees_is_initialized = true;
  }

  void set_in_degrees() {
    assert(in_boundaries_is_initialized);
    assert(in_offsets_is_initialized);

    loop::each_numa_node(
        in_boundaries, [&](u32 block_id, u32 numa_id, u32 lower, u32 upper) {
          const auto size = upper - lower;
          const auto degrees = mem::malloc<ID>(size, numa_id);
          for (u32 i = lower; i < upper - 1; ++i) {
            degrees[i - lower] =
                in_offsets(i + 1, block_id) - in_offsets(i, block_id);
          }
          degrees[size - 1] =
              (in_boundaries[num_threads] != upper // not last block
                   ? in_offsets(upper, block_id + 1)
                   : in_offsets(upper, block_id)) // cap is in curr block
              - in_offsets(upper - 1, block_id);

          in_degrees.add(degrees, size);
        });

    in_degrees_is_initialized = true;
  }

  void set_out_indices() {
    assert(out_boundaries_is_initialized);
    assert(out_offsets_is_initialized);

    // FIXME
    if (num_numa_nodes == 1) {
      out_indices.add(tmp_out_indices, num_edges);
    } else {
      loop::each_numa_node(
          out_boundaries, out_offsets,
          [&](u32 block_id, u32 numa_id, u32 start, u32 end) {
            const auto size = end - start;
            const auto indices = mem::malloc<u32>(size, numa_id);
            std::memcpy(indices, tmp_out_indices + start, size * sizeof(u32));
            out_indices.add(indices, size);
          });

      mem::free(tmp_out_indices, num_edges);
    }
    out_indices_is_initialized = true;
  }

  void set_in_indices() {
    assert(in_boundaries_is_initialized);
    assert(in_offsets_is_initialized);

    // FIXME
    if (num_numa_nodes == 1) {
      in_indices.add(tmp_in_indices, num_edges);
    } else {
      loop::each_numa_node(
          in_boundaries, in_offsets,
          [&](u32 block_id, u32 numa_id, u32 start, u32 end) {
            const auto size = end - start;
            const auto indices = mem::malloc<u32>(size, numa_id);
            std::memcpy(indices, tmp_in_indices + start, size * sizeof(u32));
            in_indices.add(indices, size);
          });

      mem::free(tmp_in_indices, sizeof(ID) * num_edges);
    }
    in_indices_is_initialized = true;
  }

  void set_out_neighbor() {
    assert(out_indices_is_initialized);
    assert(out_offsets_is_initialized);

    loop::each_numa_node(
        out_boundaries, [&](u32 block_id, u32 numa_id, u32 lower, u32 upper) {
          auto out_neighbor = colle::make_numa_vector<ID *>(numa_id);
          out_neighbor->reserve(upper - lower);
          for (ID i = lower; i < upper; ++i) {
            out_neighbor->emplace_back(
                &out_indices(out_offsets(i, block_id), block_id));
          }
          out_neighbors.add(std::move(*out_neighbor));
        });

    out_neighbors_is_initialized = true;
  }

  void set_in_neighbors() {
    assert(in_indices_is_initialized);
    assert(in_offsets_is_initialized);

    loop::each_numa_node(
        in_boundaries, [&](u32 block_id, u32 numa_id, u32 lower, u32 upper) {
          auto in_neighbor = colle::make_numa_vector<ID *>(numa_id);
          in_neighbor->reserve(upper - lower);
          for (ID i = lower; i < upper; ++i) {
            in_neighbor->emplace_back(
                &in_indices(in_offsets(i, block_id), block_id));
          }
          in_neighbors.add(std::move(*in_neighbor));
        });

    in_neighbors_is_initialized = true;
  }

  void set_forward_indices() {
    assert(out_boundaries_is_initialized);
    assert(out_offsets_is_initialized);
    assert(out_degrees_is_initialized);
    assert(out_neighbors_is_initialized);
    assert(in_offsets_is_initialized);

    forward_indices = mem::malloc<ID>(num_edges); // TODO: should be numa-local

    auto counts = std::vector<ID>(num_vertices, 0);
    loop::each_thread(out_boundaries, [&](u32 block_id, u32 numa_id,
                                          u32 thread_id, u32 lower, u32 upper) {
      for (ID src = lower; src < upper; ++src) {
        const auto neighbor = out_neighbors(src, block_id);
        for (ID i = 0, end = out_degrees(src, block_id); i < end; ++i) {
          const auto dst = neighbor[i];
          forward_indices[out_offsets(src, block_id) + i] =
              in_offsets(dst) + counts[dst];
          counts[dst]++;
        }
      }
    });

    forward_indices_is_initialized = true;
  }

  void set_v_data(bool allow_overwrite = false) {
    assert(out_boundaries_is_initialized);
    // assert(in_boundaries_is_initialized);

    if (allow_overwrite) {
      v_data = *(new colle::DiscreteArray<VData>());
    }

    // TODO: consider both out and in boundaries (?)
    // If readonly, it should be allowed that duplicate vertex data
    // And should be allocated on each numa node
    loop::each_numa_node(out_boundaries,
                         [&](u32 block_id, u32 numa_id, u32 lower, u32 upper) {
                           const auto size = upper - lower;
                           v_data.add(mem::malloc<VData>(size, numa_id), size);
                         });
  }

  void set_e_data(bool allow_overwirte = false) {
    assert(out_boundaries_is_initialized);
    assert(out_offsets_is_initialized);
    // assert(in_boundaries_is_initialized);
    // assert(in_offsets_is_initialized);

    if (allow_overwirte) {
      e_data = *(new colle::DiscreteArray<EData>());
    }

    // TODO: consider both out and in boundaries (?)
    // If readonly, it should be allowed that duplicate edge data
    // And should be allocated on each numa node
    loop::each_numa_node(out_boundaries, out_offsets,
                         [&](u32 block_id, u32 numa_id, u32 start, u32 end) {
                           const auto size = end - start;
                           e_data.add(mem::malloc<EData>(size, numa_id), size);
                         });
  }

  static void Next(_Graph &prev, _Graph &curr) {
    // TODO
    std::swap(prev.v_data, curr.v_data);
    std::swap(prev.e_data, curr.e_data);
  }

  //        template<class Executor>
  //        static _Graph FromEdgeListWithExecutor() {
  //            auto hoge = Executor::malloc();
  //            return Empty(); //
  //        }

  // TODO: poor performance
  // *require packed index* (process in preprocessing)
  static _Graph FromEdgeList(const std::pair<ID, ID> *edge_list, size_t len) {
    assert(len > 0 && edge_list != nullptr);

    using VVType = std::pair<ID, ID>;
    auto vec = std::vector<VVType>(edge_list, edge_list + len); // [(from, to)]

    // outbound
    std::sort(begin(vec), end(vec), [](const VVType &l, const VVType &r) {
      return l.second < r.second;
    });
    auto tmp_max = vec.back().second;
    std::stable_sort(
        begin(vec), end(vec),
        [](const VVType &l, const VVType &r) { return l.first < r.first; });

    auto num_vertices = std::max(tmp_max, vec.back().first) + 1; // 0-based
    auto num_edges = len;

    auto out_offsets = mem::malloc<ID>(num_vertices + 1); // all vertices + cap
    auto out_indices = mem::malloc<ID>(num_edges);        // all edges

    out_offsets[0] = 0u;
    auto prev_src = vec[0].first;
    for (u32 i = 0; i < num_edges; ++i) {
      auto curr = vec[i];
      // next source vertex
      if (prev_src != curr.first) {
        // loop for vertices whose degree is 0
        for (auto j = prev_src + 1; j <= curr.first; ++j) {
          out_offsets[j] = i;
        }
        prev_src = curr.first;
      }
      out_indices[i] = curr.second;
    }
    for (auto i = prev_src + 1; i <= num_vertices; ++i) {
      out_offsets[i] = num_edges;
    }

    // and vice versa (inbound)
    for (auto &p : vec) {
      std::swap(p.first, p.second);
    }
    std::sort(begin(vec), end(vec), [](const VVType &l, const VVType &r) {
      return l.second < r.second;
    });
    std::stable_sort(
        begin(vec), end(vec),
        [](const VVType &l, const VVType &r) { return l.first < r.first; });

    auto in_offsets = mem::malloc<ID>(num_vertices);
    auto in_indices = mem::malloc<ID>(num_edges);

    in_offsets[0] = 0;
    prev_src = vec[0].first;
    for (u32 i = 0; i < num_edges; ++i) {
      auto curr = vec[i];
      if (prev_src != curr.first) {
        // loop for vertices whose degree is 0
        for (auto j = prev_src + 1; j <= curr.first; ++j) {
          in_offsets[j] = i;
        }
        prev_src = curr.first;
      }
      in_indices[i] = curr.second;
    }
    for (auto i = prev_src + 1; i <= num_vertices; ++i) {
      in_offsets[i] = num_edges;
    }

    auto g = _Graph();
    g.num_vertices = num_vertices;
    g.num_edges = num_edges;
    g.tmp_out_offsets = out_offsets;
    g.tmp_out_indices = out_indices;
    g.tmp_in_offsets = in_offsets;
    g.tmp_in_indices = in_indices;
    g.set_out_boundaries();
    g.set_out_offsets();
    g.set_out_degrees();
    g.set_out_indices();
    g.set_out_neighbor();
    g.set_in_boundaries();
    g.set_in_offsets();
    g.set_in_degrees();
    g.set_in_indices();
    g.set_in_neighbors();
    g.set_forward_indices();
    g.set_v_data();
    g.set_e_data();

    return g;
  }
};
} // namespace hoshizora

#endif // HOSHIZORA_GRAPH_H
