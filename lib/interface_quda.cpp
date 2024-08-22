#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>
#include <complex.h>

#include <quda.h>
#include <quda_internal.h>
#include <device.h>
#include <timer.h>
#include <comm_quda.h>
#include <tune_quda.h>
#include <algorithm>
#include <random_quda.h>
#include <mpi_comm_handle.h>

#define MAX(a,b) ((a)>(b)? (a):(b))
#define TDIFF(a,b) (b.tv_sec - a.tv_sec + 0.000001*(b.tv_usec - a.tv_usec))

// define newQudaGaugeParam() and newQudaInvertParam()
#define INIT_PARAM
#include "check_params.h"
#undef INIT_PARAM

// define (static) checkGaugeParam() and checkInvertParam()
#define CHECK_PARAM
#include "check_params.h"
#undef CHECK_PARAM
void checkBLASParam(QudaBLASParam &param) { checkBLASParam(&param); }

// define printQudaGaugeParam() and printQudaInvertParam()
#define PRINT_PARAM
#include "check_params.h"
#undef PRINT_PARAM

using namespace quda;
#include <blas_lapack.h>
#include <examples.h>


// Mapped memory buffer used to hold unitarization failures
static int *num_failures_h = nullptr;
static int *num_failures_d = nullptr;

static bool initialized = false;

//!< Profiler for initQuda
static TimeProfile profileInit("initQuda");

//!< Profiler for scaling test
static TimeProfile profileScaleVector("scaleVector");

//!< Profiler for reduction test
static TimeProfile profileReduceVector("reduceVector");

//!< Profiler for GEMM and other BLAS
static TimeProfile profileBLAS("blasQuda");
TimeProfile &getProfileBLAS() { return profileBLAS; }

//!< Profiler for endQuda
static TimeProfile profileEnd("endQuda");

//!< Profiler for toal time spend between init and end
static TimeProfile profileInit2End("initQuda-endQuda",false);

static bool enable_profiler = false;
static bool do_not_profile_quda = false;

static void profilerStart(const char *f)
{
  static std::vector<int> target_list;
  static bool enable = false;
  static bool init = false;
  if (!init) {
    char *profile_target_env = getenv("QUDA_ENABLE_TARGET_PROFILE"); // selectively enable profiling for a given solve

    if ( profile_target_env ) {
      std::stringstream target_stream(profile_target_env);

      int target;
      while(target_stream >> target) {
       target_list.push_back(target);
       if (target_stream.peek() == ',') target_stream.ignore();
     }

     if (target_list.size() > 0) {
       std::sort(target_list.begin(), target_list.end());
       target_list.erase( unique( target_list.begin(), target_list.end() ), target_list.end() );
       warningQuda("Targeted profiling enabled for %lu functions\n", target_list.size());
       enable = true;
     }
   }

    char* donotprofile_env = getenv("QUDA_DO_NOT_PROFILE"); // disable profiling of QUDA parts
    if (donotprofile_env && (!(strcmp(donotprofile_env, "0") == 0)))  {
      do_not_profile_quda=true;
      printfQuda("Disabling profiling in QUDA\n");
    }
    init = true;
  }

  static int target_count = 0;
  static unsigned int i = 0;
  if (do_not_profile_quda){
    device::profile::stop();
    printfQuda("Stopping profiling in QUDA\n");
  } else {
    if (enable) {
      if (i < target_list.size() && target_count++ == target_list[i]) {
        enable_profiler = true;
        printfQuda("Starting profiling for %s\n", f);
        device::profile::start();
        i++; // advance to next target
    }
  }
}
}

static void profilerStop(const char *f) {
  if (do_not_profile_quda) {
    device::profile::start();
  } else {

    if (enable_profiler) {
      printfQuda("Stopping profiling for %s\n", f);
      device::profile::stop();
      enable_profiler = false;
    }
  }
}


namespace quda {
  void printLaunchTimer();
}

void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[], FILE *outfile)
{
  setVerbosity(verbosity);
  setOutputPrefix(prefix);
  setOutputFile(outfile);
}


typedef struct {
  int ndim;
  int dims[QUDA_MAX_DIM];
} LexMapData;

/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */
static int lex_rank_from_coords(const int *coords, void *fdata)
{
  auto *md = static_cast<LexMapData *>(fdata);

  int rank = coords[0];
  for (int i = 1; i < md->ndim; i++) {
    rank = md->dims[i] * rank + coords[i];
  }
  return rank;
}

#ifdef QMP_COMMS
/**
 * For QMP, we use the existing logical topology if already declared.
 */
static int qmp_rank_from_coords(const int *coords, void *) { return QMP_get_node_number_from(coords); }
#endif

// Provision for user control over MPI comm handle
// Assumes an MPI implementation of QMP

#if defined(QMP_COMMS) || defined(MPI_COMMS)
MPI_Comm MPI_COMM_HANDLE_USER;
static bool user_set_comm_handle = false;
#endif

#if defined(QMP_COMMS) || defined(MPI_COMMS)
void setMPICommHandleQuda(void *mycomm)
{
  MPI_COMM_HANDLE_USER = *((MPI_Comm *)mycomm);
  user_set_comm_handle = true;
}
#else
void setMPICommHandleQuda(void *) { }
#endif

static bool comms_initialized = false;

void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata)
{
  if (comms_initialized) return;

  if (nDim != 4) {
    errorQuda("Number of communication grid dimensions must be 4");
  }

  LexMapData map_data;
  if (!func) {

#if QMP_COMMS
    if (QMP_logical_topology_is_declared()) {
      if (QMP_get_logical_number_of_dimensions() != 4) {
        errorQuda("QMP logical topology must have 4 dimensions");
      }
      for (int i=0; i<nDim; i++) {
        int qdim = QMP_get_logical_dimensions()[i];
        if(qdim != dims[i]) {
          errorQuda("QMP logical dims[%d]=%d does not match dims[%d]=%d argument", i, qdim, i, dims[i]);
        }
      }
      fdata = nullptr;
      func = qmp_rank_from_coords;
    } else {
      warningQuda("QMP logical topology is undeclared; using default lexicographical ordering");
#endif

      map_data.ndim = nDim;
      for (int i=0; i<nDim; i++) {
        map_data.dims[i] = dims[i];
      }
      fdata = (void *) &map_data;
      func = lex_rank_from_coords;

#if QMP_COMMS
    }
#endif

  }

#if defined(QMP_COMMS) || defined(MPI_COMMS)
  comm_init(nDim, dims, func, fdata, user_set_comm_handle, (void *)&MPI_COMM_HANDLE_USER);
#else
  comm_init(nDim, dims, func, fdata);
#endif

  comms_initialized = true;
}


static void init_default_comms()
{
#if defined(QMP_COMMS)
  if (QMP_logical_topology_is_declared()) {
    int ndim = QMP_get_logical_number_of_dimensions();
    const int *dims = QMP_get_logical_dimensions();
    initCommsGridQuda(ndim, dims, nullptr, nullptr);
  } else {
    errorQuda("initQuda() called without prior call to initCommsGridQuda(),"
        " and QMP logical topology has not been declared");
  }
#elif defined(MPI_COMMS)
  errorQuda("When using MPI for communications, initCommsGridQuda() must be called before initQuda()");
#else // single-GPU
  const int dims[4] = {1, 1, 1, 1};
  initCommsGridQuda(4, dims, nullptr, nullptr);
#endif
}


extern char* gitversion;

/*
 * Set the device that QUDA uses.
 */
void initQudaDevice(int dev)
{
  //static bool initialized = false;
  if (initialized) return;
  initialized = true;

  profileInit2End.TPSTART(QUDA_PROFILE_TOTAL);
  auto profile = pushProfile(profileInit);
  profileInit.TPSTART(QUDA_PROFILE_INIT);

#ifdef GITVERSION
  logQuda(QUDA_SUMMARIZE, "QUDA %s (git %s)\n", get_quda_version().c_str(), gitversion);
#else
  logQuda(QUDA_SUMMARIZE, "QUDA %s\n", get_quda_version().c_str());
#endif

#ifdef MULTI_GPU
  if (dev < 0) {
    if (!comms_initialized) {
      errorQuda("initDeviceQuda() called with a negative device ordinal, but comms have not been initialized");
    }
    dev = comm_gpuid();
  }
#else
  if (dev < 0 || dev >= 16) errorQuda("Invalid device number %d", dev);
#endif

  device::init(dev);
  profileInit.TPSTOP(QUDA_PROFILE_INIT);
}

/*
 * Any persistent memory allocations that QUDA uses are done here.
 */
void initQudaMemory()
{
  auto profile = pushProfile(profileInit);
  profileInit.TPSTART(QUDA_PROFILE_INIT);

  if (!comms_initialized) init_default_comms();

  device::create_context();

  loadTuneCache();

  // initalize the memory pool allocators
  pool::init();

  //createDslashEvents();

  blas_lapack::native::init();

  num_failures_h = static_cast<int *>(mapped_malloc(sizeof(int)));
  num_failures_d = static_cast<int *>(get_mapped_device_pointer(num_failures_h));

  //for (int d=0; d<4; d++) R[d] = 2 * (redundant_comms || commDimPartitioned(d));

  profileInit.TPSTOP(QUDA_PROFILE_INIT);
}

void updateR()
{
  //for (int d=0; d<4; d++) R[d] = 2 * (redundant_comms || commDimPartitioned(d));
}

void initQuda(int dev)
{
  // initialize communications topology, if not already done explicitly via initCommsGridQuda()
  if (!comms_initialized) init_default_comms();

  // set the device that QUDA uses
  initQudaDevice(dev);

  // set the persistant memory allocations that QUDA uses (Blas, streams, etc.)
  initQudaMemory();
}

void vector_scale(double sf, void *data, unsigned long long int N, QudaPrecision prec) {

  auto profile = pushProfile(profileScaleVector);
  pushOutputPrefix("performVectorScale: ");

  size_t bytes = 0;
  switch(prec) {
  case QUDA_DOUBLE_PRECISION: bytes = sizeof(double); break;
  case QUDA_SINGLE_PRECISION: bytes = sizeof(float); break;
  default: errorQuda("Unsupported precision %d", prec);
  }  
  
  profileScaleVector.TPSTART(QUDA_PROFILE_H2D);
  quda_ptr ptr(QUDA_MEMORY_DEVICE, N*bytes, false);  
  qudaMemcpy(ptr.data(), data, N*bytes, qudaMemcpyHostToDevice);
  profileScaleVector.TPSTOP(QUDA_PROFILE_H2D);
  
  scaleVector(ptr.data(), N, sf, prec);
  
  profileScaleVector.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(data, ptr.data(), N*bytes, qudaMemcpyDeviceToHost);
  profileScaleVector.TPSTOP(QUDA_PROFILE_D2H);
  
  popOutputPrefix();
}

double vector_reduce(void *data, unsigned long long int N, QudaPrecision prec) {

  auto profile = pushProfile(profileReduceVector);
  pushOutputPrefix("performVectorReduction: ");

  size_t bytes = 0;
  switch(prec) {
  case QUDA_DOUBLE_PRECISION: bytes = sizeof(double); break;
  case QUDA_SINGLE_PRECISION: bytes = sizeof(float); break;
  default: errorQuda("Unsupported precision %d", prec);
  }  
  
  profileReduceVector.TPSTART(QUDA_PROFILE_H2D);
  quda_ptr ptr(QUDA_MEMORY_DEVICE, N*bytes, false);  
  qudaMemcpy(ptr.data(), data, N*bytes, qudaMemcpyHostToDevice);
  profileReduceVector.TPSTOP(QUDA_PROFILE_H2D);

  double sum = reduceVector(ptr.data(), N, prec);
  
  profileReduceVector.TPSTART(QUDA_PROFILE_D2H);
  qudaMemcpy(data, ptr.data(), N*bytes, qudaMemcpyDeviceToHost);
  profileReduceVector.TPSTOP(QUDA_PROFILE_D2H);
  
  popOutputPrefix();
  return sum;
}

void endQuda(void)
{
  if (!initialized) return;

  {
    auto profile = pushProfile(profileEnd);


    blas_lapack::generic::destroy();
    blas_lapack::native::destroy();

    pool::flush_pinned();
    pool::flush_device();

    host_free(num_failures_h);
    num_failures_h = nullptr;
    num_failures_d = nullptr;

    saveTuneCache();
    saveProfile();

    initialized = false;

    assertAllMemFree();
    device::destroy();

    comm_finalize();
    comms_initialized = false;
  }

  profileInit2End.TPSTOP(QUDA_PROFILE_TOTAL);

  // print out the profile information of the lifetime of the library
  if (getVerbosity() >= QUDA_SUMMARIZE) {
    profileInit.Print();
    profileScaleVector.Print();
    profileReduceVector.Print();
    profileBLAS.Print();
    profileEnd.Print();

    profileInit2End.Print();
    TimeProfile::PrintGlobal();

    printLaunchTimer();
    printAPIProfile();

    printfQuda("\n");
    printPeakMemUsage();
    printfQuda("\n");
  }
}


