#include <cassert>

#include <opencv2/gapi/sycl/gsyclkernel.hpp>

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
