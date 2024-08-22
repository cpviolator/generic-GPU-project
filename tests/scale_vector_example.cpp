#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <complex>

#include <inttypes.h>

#include <test.h>
#include <blas_reference.h>
#include <misc.h>

// In a typical application, quda.h is the only QUDA header required.
#include <quda.h>

namespace quda {
  extern void setTransferGPU(bool);
}

template <typename Float> void populate(void *data, unsigned long long int N) {
  Float *p = (Float*)data;
  for (unsigned long long int i=0; i<N; i++) p[i] = (1.0 * i) / N;  
}

template <typename Float> double sum(void *data, unsigned long long int N) {
  double sum = 0;
  Float *p = (Float*)data;
  for(unsigned long long int i=0; i<N; i++) sum += p[i] * p[i];
  return sum;
}

void display_test_info()
{
  printfQuda("running the following test:\n");
  printfQuda("Scale Vector test\n");
  printfQuda("Grid partition info:     X  Y  Z  T\n");
  printfQuda("                         %d  %d  %d  %d\n", dimPartitioned(0), dimPartitioned(1), dimPartitioned(2),
             dimPartitioned(3));
}

int main(int argc, char **argv) {

  auto app = make_app();
  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }
  
  double result_pre = 0;
  double result_post = 0;
  unsigned long long int N = n_elems;
  double scaling_factor = 3.14159;
  size_t bytes = 0;
  
  if(prec == QUDA_DOUBLE_PRECISION) bytes = sizeof(double);
  else if(prec == QUDA_SINGLE_PRECISION) bytes = sizeof(float);
  else errorQuda("Unsupported precision %d ", prec);
  void *data = malloc(N * bytes);
  
  switch(prec) {
  case QUDA_DOUBLE_PRECISION: 
    populate<double>(data, N);
    result_pre = sum<double>(data, N);
    break;
  case QUDA_SINGLE_PRECISION: 
    populate<float>(data, N);
    result_pre = sum<float>(data, N);
    break;
  default:
    errorQuda("Unsupported precision %d ", prec);
  }
  result_pre = sqrt(result_pre);
  
  // initialize QMP/MPI, QUDA comms grid and RNG (host_utils.cpp)
  initComms(argc, argv, gridsize_from_cmdline);  
  initQuda(device_ordinal);
  display_test_info();
  
  // Call GPU interface function 
  vector_scale(scaling_factor, (void*)data, N, prec);
  
  // Clean up.
  endQuda();
  
  // Checks
  switch(prec) {
  case QUDA_DOUBLE_PRECISION: result_post = sum<double>(data, N); break;
  case QUDA_SINGLE_PRECISION: result_post = sum<float>(data, N); break;
  default:
    errorQuda("Unsupported precision %d", prec);
  }
  result_post = sqrt(result_post);

  printfQuda("The PRE  scale norm is %f\n", result_pre);
  printfQuda("The POST scale norm is %f\n", result_post);
  double dev = (scaling_factor - result_post/result_pre) / result_pre;
  printfQuda("Scaling deviation: [scaling_factor - norm(post)/norm(pre)] / norm(pre) = %f\n", dev);

  free(data);
  
  return 0;
}
