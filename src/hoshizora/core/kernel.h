#ifndef HOSHIZORA_KERNEL_H
#define HOSHIZORA_KERNEL_H

#include "hoshizora/core/includes.h"

namespace hoshizora {
template <class Graph> struct Kernel {
  using EData = typename Graph::_EData;
  using VData = typename Graph::_VData;
  using ID = typename Graph::_ID;

  virtual VData init(const ID src, const Graph &graph) const = 0;

  virtual EData scatter(const ID src, const ID dst, const VData v_val,
                        Graph &graph) = 0;

  virtual EData gather(const ID src, const ID dst, const EData prev_val,
                       const EData curr_val, const Graph &graph) const = 0;

  virtual VData zero(const ID dst, const Graph &graph) const = 0;

  virtual VData sum(const ID src, const ID dst, const VData v_val,
                    const EData e_val, Graph &graph) = 0;

  virtual VData apply(const ID dst, const VData prev_val, const VData curr_val,
                      const Graph &graph) const = 0;

  virtual std::vector<std::string> result(const Graph &graph) const = 0;
};
} // namespace hoshizora

#endif // HOSHIZORA_KERNEL_H
