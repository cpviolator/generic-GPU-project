#include <tunable_nd.h>
#include <tunable_reduction.h>
#include <instantiate.h>
#include <device_vector.h>
#include <kernels/scale_vector.cuh>

namespace quda {

  // Scaled Vector (example code)
  template <typename Float> class ScaleVector : TunableKernel1D {

  protected:
    unsigned long long int N;
    void *data;
    double sf;
    unsigned int minThreads() const { return N; }

    device_vector<Float> _backup_vector;
    Float *_backup_ptr = nullptr;
    
  public:
    ScaleVector(const QudaPrecision prec, void *data, const unsigned long long int N, const double sf) : 
      TunableKernel1D(N, QUDA_CUDA_FIELD_LOCATION),
      N(N),
      data(data),
      sf(sf)
    {
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      launch<VectorScale>(tp, stream, vsArg<Float>(N, (Float*)data, sf));
    }

    void preTune()
    {
      _backup_vector.resize(N);
      _backup_vector.from_device((Float*)data);
      _backup_ptr = (Float*)data;
      data = (void*)_backup_vector.data();
    }

    void postTune() { data = _backup_ptr; }
    
    // Artificially expand the flop count by nultiplying in kernel
    // by sf and 1/sf 16 times.
    long long flops() const { return 16 * 2 * N; }    

    // Amount of H2D ddata transfer in kernel.
    long long bytes() const { return N * sizeof(Float); }
    
  };
  
  void scaleVector(void *data, const unsigned long long int N, const double sf, const QudaPrecision prec)
  {
    getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    instantiatePrecisionTEST<ScaleVector>(prec, data, N, sf);
    getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
  }

} // namespace quda
