#ifndef OPENCV_GAPI_GSYCLKERNEL_HPP
#define OPENCV_GAPI_GSYCLKERNEL_HPP

#include <vector>
#include <functional>
#include <map>
#include <unordered_map>

#include <opencv2/core/mat.hpp>
#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/garg.hpp>

// FIXME: namespace scheme for backends?
namespace cv {

namespace gimpl
{
  // Forward-declare an internal class
  class GSYCLExecutable;
} // namespace gimpl

namespace gapi
{
/**
 * @brief This namespace contains G-API SYCL backend functions, structures, and symbols.
 */
namespace sycl
{
  /**
   * \addtogroup gapi_std_backends G-API Standard Backends
   * @{
   */
  /**
   * @brief FIXME: relevant dexoygen documentation for sycl backend
   * @sa gapi_std_backends
   */
  GAPI_EXPORTS cv::gapi::GBackend backend();
  /** @} */
} // namespace sycl
} // namespace gapi

// Represents arguments which are passed to a wrapped SYCL function
// FIXME: put into detail?
class GAPI_EXPORTS GSYCLContext
{
public:
  // Generic accessor API
  template<typename T>
  const T& inArg(int input) { return m_args.at(input).get<T>(); }

  // Syntax sugar
  const cv::UMat& inMat(int input);
  cv::UMat& outMatR(int output); // FIXME: Figure out if we're sticking with UMats
                                 // and where those changes need to be made
                                 //

  const cv::Scalar& inVal(int input);
  cv::Scalar& outValR(int output); // FIXME: Avoid cv::Scalar s = stx.outValR()
  template<typename T> std::vector<T>& outVecR(int output) // FIXME: the same issue
  {
    return outVecRef(output).wref<T>();
  }
  template<typename T> T& outOpaqueR(int output) // FIXME: the same issue
  {
    return outOpaqueRef(output).wref<T>();
  }

protected:
  detail::VectorRef& outVecRef(int output);
  detail::OpaqueRef& outOpaqueRef(int output);

  std::vector<GArg> m_args;
  std::unordered_map<std::size_t, SRunArgP> m_results;

  friend class gimpl::GSYCLExecutable;
};

class GAPI_EXPORTS GSYCLKernel
{
public:
  // This function is kernel's execution entry point (does the processing work)
  using F = std::function<void(GSYCLContext &)>;

  GSYCLKernel();
  explicit GSYCLKernel(const F& f);

  void apply(GSYCLContext &ctx);

protected:
  F m_f;
};
