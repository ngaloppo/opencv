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

// FIXME: These are not included with OpenCV by default
// This is protected by HAVE_SYCL in gsyclbackend.hpp, is this included anywhere else?
#include <CL/sycl.hpp>
#include <CL/sycl/backend/opencl.hpp> 

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
    // TODO: Figure out where GSYCLContext values are initialized
    class GAPI_EXPORTS GSYCLContext
    {
    public:
        GSYCLContext();

        // Generic accessor API
        template<typename T>
        const T& inArg(int input) { return m_args.at(input).get<T>(); }

        // Syntax sugar
        const sycl::buffer<float, 1> inMat(int input);

        sycl::buffer<float, 1>& outMatR(int output); // TODO: Figure out if we're sticking with UMats
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
        // SYCL specific values
        // TODO: Determine when these get assigned
        sycl::queue& m_queue;
        sycl::context& m_context;

        void initSYCLContext();

        detail::VectorRef& outVecRef(int output);
        detail::OpaqueRef& outOpaqueRef(int output);

        std::vector<GArg> m_args;
        std::unordered_map<std::size_t, GRunArgP> m_results;

        friend class gimpl::GSYCLExecutable;
    };

    class GAPI_EXPORTS GSYCLKernel
    {
    public:
        // This function is kernel's execution entry point (does the processing work)
        using F = std::function<void(GSYCLContext&)>;

        GSYCLKernel();
        explicit GSYCLKernel(const F& f);

        void apply(GSYCLContext& ctx);

    protected:
        F m_f;
    };

    // FIXME: This is an ugly ad-hoc implementation. TODO: refactor
    // TODO: Figure out the difference between the gtyped input wrappers and the cv typed ones
    namespace detail
    {
        template<class T> struct sycl_get_in;
        template<> struct sycl_get_in<cv::GMat>
        {
            // static sycl::Buffer
        };
    } // namespace detail

    template<class Impl, class K>
    class GSYCLKernelImpl : public cv::detail::SYCLCallHelper<Impl, typename K::InArgs, typename K::OutArgs>,
        public cv::detail::KernelTag
    {
        // FIXME: Taken from OCL implementation, figure out if necessary
        using P = detail::SYCLCallHelper<Impl, typename K::InArgs, typename K::OutArgs>;

    public:
        using API = K;

        static cv::gapi::GBackend backend() { return cv::gapi::sycl::backend(); }
        static cv::GSYCLKernel    kernel() { return GSYCLKernel(&P::call); }
    };

#define GAPI_SYCL_KERNEL(Name, API) struct Name: public cv::GSYCLKernelImpl<Name, API>

} // namespace cv

#endif // OPENCV_GAPI_GSYCLKERNEL_HPP
