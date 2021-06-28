#ifdef HAVE_SYCL

#ifndef OPENCV_GAPI_GSYCLBACKEND_HPP
#define OPENCV_GAPI_GSYCLBACKEND_HPP

//includes go here
// opencl/sycl backend includes
//#include <CL/sycl.hpp>
//#include <CL/sycl/backend/opencl.hpp>

namespace cv { namespace gimpl {

struct SYCLUnit
{
  static const char *name() { return "SYCLKernel"; }
  GSYCLKernel k;
};

class GSYCLExecutable final: public GIslandExecutable
{
public:

  GSYCLExecutable(const ade::Graph  &graph,
                  const std::vector<ade::NodeHandle> &nodes);

  virtual void run(std::vector<InObj> &&input_objs,
                   std::vector<OutObj> &&ouput_objs) override;
};
}}

#endif // OPENCV_GAPI_GSYCLBACKEND_HPP

#endif // HAVE_SYCL
