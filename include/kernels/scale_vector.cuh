#pragma once

#include <comm_quda.h>
#include <kernel.h>
#include <register_traits.h>

namespace quda {
  
  template <typename Float> struct vsArg : kernel_param<> {
    using real = typename mapper<Float>::type;    
    Float *data;
    double scale_factor;
    
    vsArg(unsigned long long int N, Float *data, double scale_factor) :
      kernel_param(dim3(N, 1, 1)),
      data(data),
      scale_factor(scale_factor)
    {
    }
  };

  template <typename Arg> struct VectorScale {
    const Arg &arg;
    constexpr VectorScale(const Arg &arg) : arg(arg) {}
    static constexpr const char *filename() { return KERNEL_FILE; }
    
    __device__ __host__ inline void operator()(int idx)
    {
      double sf = arg.scale_factor;
      double inv_sf = 1.0/arg.scale_factor;
      double datum = arg.data[idx];      
      for(int i=0; i<16; i++) {
	datum *= sf;
	datum *= inv_sf;
      }
      datum *= sf;
      arg.data[idx] = datum;
    }
  };  
}
