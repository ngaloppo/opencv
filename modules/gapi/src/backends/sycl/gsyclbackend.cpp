#ifdef HAVE_SYCL

#include "precomp.hpp"

#include <ade/util/algorithm.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/sycl/gsyclbackend.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

using GSYCLModel = ade::TypedGraph
  < cv::gimpl::SYCLUnit
  , cv::gimpl::Protocol
  >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGSYCLModel = ade::ConstTypedGraph
  < cv::gimpl::SYCLUnit
  , cv::gimpl::Protocol
  >;

namespace
{
  class GSYCLBackendImpl final: public cv::gapi::Gbackend::Priv
  {
    virtual void unpackKernel(ade::Graph  &graph,
                              const ade::NodeHandle &op_node,
                              const cv::GKernelImpl &impl) override
    {
      GSYCLModel gm(graph);
      auto sycl_impl = cv::util::any_cast<cv::GSYCLKernel>(impl.opaque);
      gm.metadata(op_node).set(cv::gimpl::SYCLUnit{sycl_impl});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const cv::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
      return EPtr{new cv::gimpl::GSYCLExecutable(graph, nodes)};
    }
  };
}

cv::gapi::GBackend cv::gapi::sycl::backend()
{
  static cv::gapi::GBackend this_backend(std::make_shared<GSYCLBackendImpl>());
  return this_backend;
}

// GSYCLExecutable implementation ////////////////////
cv::gimpl::GSYCLExecutable::GSYCLExecutable(const ade::Graph &g,
                                            const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g)
{
  // Convert op list into execution script
  // Possibly add interop node here to transition from Mats to sycl buffers
  for (auto &nh : nodes)
  {
    switch (m_gm.metadata(nh).get<NodeType>().t)
    {
      case NodeType::OP: m_script.push_back({nh, GMode::collectOutputMeta(m_gm, nh)}); break;
      case NodeType::DATA:
      {
        m_dataNodes.push_back(nh);
        const auto &desc = m_gm.metadata(nh).get<Data>();
        if (desc.storage == Data::Storage::CONST_VAL)
        {
          auto rc = RcDesc{desc.rc, desc.shape, desc.ctor};
          magazine::bindInArg(m_res, rc, m_gm.metadata(nh).get<ConstValue>().arg);
        }
        //This is where internal mat allocation happens
        //need to figure out how to use buffers instead
        if (desc.storage == Data::Storage::INTERNAL && desc.shape == GShape::GMAT)
        {
          const auto mat_desc = util::get<cv::GMatDesc>(desc.meta);
          //auto& mat = m_res.slot<cv::Mat>()[desc.rc];
          //createMat(mat_desc, mat);
        }
        break;
      }
      default: util::throw_error(std::logic_error("Unsupported NodeType type"));
    }
  }
}

#endif // HAVE_SYCL
