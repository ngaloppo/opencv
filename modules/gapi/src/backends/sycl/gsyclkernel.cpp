#include <cassert>

#include <opencv2/gapi/sycl/gsyclkernel.hpp>

// Ugly interoperability implementation
// FIXME: Figure out sycl buffer construction from native gapi data structures
// FIXME: Figure out type logic, buffer is template class but UMat is not
// FIXME: Figure out optimizations such as UMat interop when preceeding nodes use ocl backend
const sycl::buffer<float, 1> cv::GSYCLContext::inMat(int input)
{
    auto inArg = inArg<cv::UMat>(input);
    cl_mem ocl_handleIn = static_cast<cl_mem>(inArg.handle(cv::ACCESS_RW));
    return sycl::make_buffer<sycl::backend::opencl, T>(ocl_handleIn, m_context);
}

// FIXME: Do we return buffers or UMats? How do we control this based on graph topology?
cv::UMat& cv::GSYCLContext::outMatR(int output)
{
    return (*(util::get<cv::UMat*>(m_results.at(output))));
}

const cv::Scalar& cv::GSYCLContext::inVal(int input)
{
  return inArg<cv::Scalar>(input);
}

cv::Scalar& cv::GSYCLContext::outValR(int output)
{
  return *util::get<cv::Scalar*>(m_results.at(output));
}

cv::GSYCLKernel::GSYCLKernel()
{
}

cv::GSYCLKernel::GSYCLKernel(const GSYCLKernel::F &f)
  : m_f(f)
{
}

void cv::GSYCLKernel::apply(GSYCLContext &ctx)
{
  CV_Assert(m_f);
  m_f(ctx);
}
