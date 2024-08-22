#include <limits>
#include <complex>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <short.h>

#include <comm_quda.h>

// This contains the appropriate ifdef guards already
#include <mpi_comm_handle.h>

#include <util_quda.h>
#include <host_utils.h>
#include <command_line_params.h>

#include <misc.h>
#include <qio_field.h>

template <typename T> using complex = std::complex<T>;

#define XUP 0
#define YUP 1
#define ZUP 2
#define TUP 3

int Z[4];
int V;
int Vh;
int Vs_x, Vs_y, Vs_z, Vs_t;
int Vsh_x, Vsh_y, Vsh_z, Vsh_t;
int faceVolume[4];

// extended volume, +4
int E1, E1h, E2, E3, E4;
int E[4];
int V_ex, Vh_ex;

int Ls;
int V5;
int V5h;
double kappa5;

extern float fat_link_max;

// Set some local QUDA precision variables
QudaPrecision local_prec = QUDA_DOUBLE_PRECISION;
QudaPrecision &cpu_prec = local_prec;
QudaPrecision &cuda_prec = prec;
QudaPrecision &cuda_prec_sloppy = prec_sloppy;
QudaPrecision &cuda_prec_refinement_sloppy = prec_refinement_sloppy;
QudaPrecision &cuda_prec_precondition = prec_precondition;
QudaPrecision &cuda_prec_ritz = prec_ritz;

size_t host_gauge_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_spinor_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
size_t host_clover_data_type_size = (cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

void setQudaPrecisions()
{
  if (prec_sloppy == QUDA_INVALID_PRECISION) prec_sloppy = prec;
  if (prec_precondition == QUDA_INVALID_PRECISION) prec_precondition = prec_sloppy;
  if (prec_null == QUDA_INVALID_PRECISION) prec_null = prec_precondition;
}

void initComms(int argc, char **argv, std::array<int, 4> &commDims) { initComms(argc, argv, commDims.data()); }

#if defined(QMP_COMMS) || defined(MPI_COMMS)
void initComms(int argc, char **argv, int *const commDims)
#else
void initComms(int, char **, int *const commDims)
#endif
{
  if (getenv("QUDA_TEST_GRID_SIZE")) { get_size_from_env(commDims, "QUDA_TEST_GRID_SIZE"); }
  if (getenv("QUDA_TEST_GRID_PARTITION")) { get_size_from_env(grid_partition.data(), "QUDA_TEST_GRID_PARTITION"); }

#if defined(QMP_COMMS)
  QMP_thread_level_t tl;
  QMP_init_msg_passing(&argc, &argv, QMP_THREAD_SINGLE, &tl);

  // make sure the QMP logical ordering matches QUDA's
  if (rank_order == 0) {
    int map[] = {3, 2, 1, 0};
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  } else {
    int map[] = {0, 1, 2, 3};
    QMP_declare_logical_topology_map(commDims, 4, map, 4);
  }
#elif defined(MPI_COMMS)
  MPI_Init(&argc, &argv);
#endif

  QudaCommsMap func = rank_order == 0 ? lex_rank_from_coords_t : lex_rank_from_coords_x;

  initCommsGridQuda(4, commDims, func, NULL);

  for (int d = 0; d < 4; d++) {
    if (dim_partitioned[d]) { quda::commDimPartitionedSet(d); }
  }

  initRand();

  printfQuda("Rank order is %s major (%s running fastest)\n", rank_order == 0 ? "column" : "row",
             rank_order == 0 ? "t" : "x");
}

void finalizeComms()
{
  quda::comm_finalize();
#if defined(QMP_COMMS)
  QMP_finalize_msg_passing();
#elif defined(MPI_COMMS)
  MPI_Finalize();
#endif
}

void initRand()
{
  int rank = 0;

#if defined(QMP_COMMS)
  rank = QMP_get_node_number();
#elif defined(MPI_COMMS)
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  srand(17 * rank + 137);
}


int dimPartitioned(int dim) { return ((gridsize_from_cmdline[dim] > 1) || dim_partitioned[dim]); }

int index_4d_cb_from_coordinate_4d(const int coordinate[4], const int dim[4])
{
  return (((coordinate[3] * dim[2] + coordinate[2]) * dim[1] + coordinate[1]) * dim[0] + coordinate[0]) >> 1;
}

void coordinate_from_shrinked_index(int coordinate[4], int shrinked_index, const int shrinked_dim[4],
                                    const int shift[4], int parity)
{
  int aux[4];
  aux[0] = shrinked_index * 2;

  for (int i = 0; i < 3; i++) { aux[i + 1] = aux[i] / shrinked_dim[i]; }

  coordinate[0] = aux[0] - aux[1] * shrinked_dim[0];
  coordinate[1] = aux[1] - aux[2] * shrinked_dim[1];
  coordinate[2] = aux[2] - aux[3] * shrinked_dim[2];
  coordinate[3] = aux[3];

  // Find the full coordinate in the shrinked volume.
  coordinate[0] += (parity + coordinate[3] + coordinate[2] + coordinate[1]) & 1;

  // if(shrinked_index == 3691) printfQuda("coordinate[0] = %d\n", coordinate[0]);
  for (int d = 0; d < 4; d++) { coordinate[d] += shift[d]; }
}

// i represents a "half index" into an even or odd "half lattice".
// when oddBit={0,1} the half lattice is {even,odd}.
//
// the displacements, such as dx, refer to the full lattice coordinates.
//
// neighborIndex() takes a "half index", displaces it, and returns the
// new "half index", which can be an index into either the even or odd lattices.
// displacements of magnitude one always interchange odd and even lattices.
//



extern "C" {
/**
   @brief Set the default ASAN options.  This ensures that QUDA just
   works when SANITIZE is enabled without requiring ASAN_OPTIONS to be
   set.  We default disable leak checking, otherwise this will cause
   ctest to fail with MPI library leaks.
 */
const char *__asan_default_options() { return "detect_leaks=0,protect_shadow_gap=0"; }
}

/**
 * For MPI, the default node mapping is lexicographical with t varying fastest.
 */

void get_size_from_env(int *const dims, const char env[])
{
  char *grid_size_env = getenv(env);
  if (grid_size_env) {
    std::stringstream grid_list(grid_size_env);

    int dim;
    int i = 0;
    while (grid_list >> dim) {
      if (i >= 4) errorQuda("Unexpected grid size array length");
      dims[i] = dim;
      if (grid_list.peek() == ',') grid_list.ignore();
      i++;
    }
  }
}

int lex_rank_from_coords_t(const int *coords, void *)
{
  int rank = coords[0];
  for (int i = 1; i < 4; i++) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

int lex_rank_from_coords_x(const int *coords, void *)
{
  int rank = coords[3];
  for (int i = 2; i >= 0; i--) { rank = gridsize_from_cmdline[i] * rank + coords[i]; }
  return rank;
}

// a+=b
template <typename Float> inline void complexAddTo(Float *a, Float *b)
{
  a[0] += b[0];
  a[1] += b[1];
}

// a = b*c
template <typename Float> inline void complexProduct(Float *a, Float *b, Float *c)
{
  a[0] = b[0] * c[0] - b[1] * c[1];
  a[1] = b[0] * c[1] + b[1] * c[0];
}

// a = conj(b)*conj(c)
template <typename Float> inline void complexConjugateProduct(Float *a, Float *b, Float *c)
{
  a[0] = b[0] * c[0] - b[1] * c[1];
  a[1] = -b[0] * c[1] - b[1] * c[0];
}

// a = conj(b)*c
template <typename Float> inline void complexDotProduct(Float *a, Float *b, Float *c)
{
  a[0] = b[0] * c[0] + b[1] * c[1];
  a[1] = b[0] * c[1] - b[1] * c[0];
}

// a += b*c
template <typename Float> inline void accumulateComplexProduct(Float *a, Float *b, Float *c, Float sign)
{
  a[0] += sign * (b[0] * c[0] - b[1] * c[1]);
  a[1] += sign * (b[0] * c[1] + b[1] * c[0]);
}

// a += conj(b)*c)
template <typename Float> inline void accumulateComplexDotProduct(Float *a, Float *b, Float *c)
{
  a[0] += b[0] * c[0] + b[1] * c[1];
  a[1] += b[0] * c[1] - b[1] * c[0];
}

template <typename Float> inline void accumulateConjugateProduct(Float *a, Float *b, Float *c, int sign)
{
  a[0] += sign * (b[0] * c[0] - b[1] * c[1]);
  a[1] -= sign * (b[0] * c[1] + b[1] * c[0]);
}


template <typename Float> static int compareFloats(Float *a, Float *b, int len, double epsilon)
{
  for (int i = 0; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon || std::isnan(diff)) {
      printfQuda("ERROR: i=%d, a[%d]=%f, b[%d]=%f\n", i, i, a[i], i, b[i]);
      return 0;
    }
  }
  return 1;
}

int compare_floats(void *a, void *b, int len, double epsilon, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    return compareFloats((double *)a, (double *)b, len, epsilon);
  else
    return compareFloats((float *)a, (float *)b, len, epsilon);
}

template <typename Float> static double compareFloats_v2(Float *a, Float *b, int len, double epsilon)
{
  double global_diff = 0.0;
  for (int i = 0; i < len; i++) {
    double diff = fabs(a[i] - b[i]);
    if (diff > epsilon || std::isnan(diff)) {
      //printfQuda("ERROR: i=%d, a[%d]=%f, b[%d]=%f\n", i, i, a[i], i, b[i]);
      return diff;
    }
    global_diff = std::max(global_diff, diff);
  }
  return global_diff;
}

// returns deviation instead of failure flag
double compare_floats_v2(void *a, void *b, int len, double epsilon, QudaPrecision precision)
{
  if (precision == QUDA_DOUBLE_PRECISION)
    return compareFloats_v2((double *)a, (double *)b, len, epsilon);
  else
    return compareFloats_v2((float *)a, (float *)b, len, epsilon);
}


// normalize the vector a
template <typename Float> static void normalize(complex<Float> *a, int len)
{
  double sum = 0.0;
  for (int i = 0; i < len; i++) sum += norm(a[i]);
  for (int i = 0; i < len; i++) a[i] /= sqrt(sum);
}

// orthogonalize vector b to vector a
template <typename Float> static void orthogonalize(complex<Float> *a, complex<Float> *b, int len)
{
  complex<double> dot = 0.0;
  for (int i = 0; i < len; i++) dot += conj(a[i]) * b[i];
  for (int i = 0; i < len; i++) b[i] -= (complex<Float>)dot * a[i];
}


void performanceStats(std::vector<double> &time, std::vector<double> &gflops, std::vector<int> &iter)
{
  /*
  auto mean_time = 0.0;
  auto mean_time2 = 0.0;
  auto mean_gflops = 0.0;
  auto mean_gflops2 = 0.0;
  auto mean_iter = 0.0;
  auto mean_iter2 = 0.0;
  // skip first solve due to allocations, potential UVM swapping overhead
  for (int i = 1; i < Nsrc; i++) {
    mean_time += time[i];
    mean_time2 += time[i] * time[i];
    mean_gflops += gflops[i];
    mean_gflops2 += gflops[i] * gflops[i];
    mean_iter += iter[i];
    mean_iter2 += iter[i] * iter[i];
  }

  auto NsrcM1 = Nsrc - 1;

  mean_time /= NsrcM1;
  mean_time2 /= NsrcM1;
  auto stddev_time = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_time2 - mean_time * mean_time)) :
                                  std::numeric_limits<double>::infinity();
  mean_gflops /= NsrcM1;
  mean_gflops2 /= NsrcM1;
  auto stddev_gflops = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_gflops2 - mean_gflops * mean_gflops)) :
                                    std::numeric_limits<double>::infinity();

  mean_iter /= NsrcM1;
  mean_iter2 /= NsrcM1;
  auto stddev_iter = NsrcM1 > 1 ? sqrt((NsrcM1 / ((double)NsrcM1 - 1.0)) * (mean_iter2 - mean_iter * mean_iter)) :
                                  std::numeric_limits<double>::infinity();
  
  printfQuda("%d solves, mean iteration count %g (stddev = %g), with mean solve time %g (stddev = %g), mean GFLOPS %g "
             "(stddev = %g) [excluding first solve]\n",
             Nsrc, mean_iter, stddev_iter, mean_time, stddev_time, mean_gflops, stddev_gflops);
  */
  
}
