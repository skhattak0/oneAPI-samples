#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_complex.hpp>

#include "exception_handler.hpp"

#include <iostream>
#include <vector>

using namespace sycl;

// This reference design uses a template-based unroller. It's also possible
// to specify this in a more concise way using a pragma. See the loop unroll
// tutorial for more information.
template <int Begin, int End>
struct Unroller {
  template <typename Action>
  static void step(const Action &action) {
    action(std::integral_constant<decltype(Begin), Begin>());
    Unroller<Begin + 1, End>::step(action);
  }
};

template <int End>
struct Unroller<End, End> {
  template <typename Action>
  static void step(const Action &action) {}
};

constexpr int InputWidth = 8;

// ensure the number of inputs is a power of 2
constexpr int NumInputsLog2 = 3;
constexpr int NumInputs = 1 << NumInputsLog2;

constexpr inline int OutputWidthOfMul(int inputWidth) { return inputWidth*2 + 1; }
constexpr inline int BitWidthAtLayer(int layer) { return (InputWidth + 1)*layer + InputWidth; }
constexpr inline int ElementsInLayer(int layer) { return NumInputs / (1 << layer); }

// a layer of the tree with width x produces a layer with width 2x+1
// this is a closed form expression to compute the final layer's width
constexpr int OutputWidth = BitWidthAtLayer(NumInputsLog2);

using InputT = ac_complex<ac_int<InputWidth, true> >;
using OutputT = ac_complex<ac_int<OutputWidth, true> >;

// the type at layer L of the tree
template <int L>
using LayerT = ac_complex<ac_int<BitWidthAtLayer(L), true> >;

constexpr access::mode kSyclRead = access::mode::read;
constexpr access::mode kSyclWrite = access::mode::write;
constexpr access::mode kSyclReadWrite = access::mode::read_write;

void test_mult(queue &Queue, const std::vector<InputT> &in, OutputT &out) {
  assert(in.size() == NumInputs);

  buffer<InputT> inputBuffer(in.begin(), in.end());
  buffer<OutputT> result(&out, 1);

  Queue.submit([&](handler &h) {
    auto x = inputBuffer.template get_access<kSyclRead>(h);
    auto res = result.template get_access<kSyclWrite>(h);
    h.single_task<class mult>([=] {
      OutputT prevBuffer[NumInputs];
      #pragma unroll
      for (int i = 0; i < NumInputs; ++i) {
        prevBuffer[i] = x[i];
      }
    
      Unroller<0, NumInputsLog2>::step([&](auto l) {
        int layerElts = ElementsInLayer(l);
        int layerPairs = layerElts / 2;
        
        #pragma unroll
        for (int i = 0; i < layerPairs; i++) {
          prevBuffer[i] = prevBuffer[2*i]*prevBuffer[2*i+1];
        }
      });

      res[0] = prevBuffer[0];
    });
  });
  Queue.wait();
}

template <typename T>
bool check_result(T expected, T found, std::string test_name) {
  if (expected != found) {
    std::cout << test_name << ":\tfailed\n";
    std::cout << "expected:\t" << expected << "\n";
    std::cout << "found:\t" << found << "\n";
    return false;
  }
  return true;
}

int main() {
#if FPGA_SIMULATOR
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  auto selector = sycl::ext::intel::fpga_selector_v;
#else  // #if FPGA_EMULATOR
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#endif

  try {
    // Create the SYCL device queue
    queue q(selector, fpga_tools::exception_handler);
    auto device = q.get_device();

    std::cout << "Running on device: "
              << device.get_info<sycl::info::device::name>().c_str()
              << std::endl;


    OutputT res = 0;
    std::vector<InputT> in = {
      InputT(10, 20),
      InputT(5, 10),
      InputT(-20, 20),
      InputT(20, 4),
      InputT(24, 3),
      InputT(4, 3),
      InputT(56, 2),
      InputT(34,24)
    };

    test_mult(q, in, res);

    
    OutputT expected = OutputT(1313, 2016);

    // Confirm the result is as expected
    if (!check_result(expected, res, "test_mult")) {
      std::cout << "FAILED\n";
      return -1;
    } else {
      std::cout << "PASSED\n";
    }    
  } catch (sycl::exception const &e) {
    // Catches exceptions in the host code
    std::cerr << "Caught a SYCL host exception:\n" << e.what() << "\n";

    // Most likely the runtime couldn't find FPGA hardware!
    if (e.code().value() == CL_DEVICE_NOT_FOUND) {
      std::cerr << "If you are targeting an FPGA, please ensure that your "
                   "system has a correctly configured FPGA board.\n";
      std::cerr << "Run sys_check in the oneAPI root directory to verify.\n";
      std::cerr << "If you are targeting the FPGA emulator, compile with "
                   "-DFPGA_EMULATOR.\n";
    }
    std::terminate();
  }

  return 0;
}
