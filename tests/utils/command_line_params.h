#pragma once

#include <CLI11.hpp>
#include <array>
#include <quda.h>

// for compatibility while porting - remove later
extern void usage(char **);

class QUDAApp : public CLI::App
{

public:
  QUDAApp(std::string app_description = "", std::string app_name = "") : CLI::App(app_description, app_name) {};

  virtual ~QUDAApp() {};

};

std::shared_ptr<QUDAApp> make_app(std::string app_description = "QUDA internal test", std::string app_name = "");
void add_comms_option_group(std::shared_ptr<QUDAApp> quda_app);
void add_testing_option_group(std::shared_ptr<QUDAApp> quda_app);

template <typename T> std::string inline get_string(CLI::TransformPairs<T> &map, T val)
{
  auto it
    = std::find_if(map.begin(), map.end(), [&val](const decltype(map.back()) &p) -> bool { return p.second == val; });
  return it->first;
}

extern int device_ordinal;
extern int rank_order;
extern bool native_blas_lapack;

extern bool verify_results;
extern bool enable_testing;

extern unsigned long long n_elems;

extern std::array<int, 4> gridsize_from_cmdline;
extern std::array<int, 4> dim_partitioned;
extern std::array<int, 4> grid_partition;

extern QudaPrecision prec;
extern QudaPrecision prec_sloppy;
extern QudaPrecision prec_refinement_sloppy;
extern QudaPrecision prec_precondition;
extern QudaPrecision prec_null;
extern QudaPrecision prec_ritz;
extern QudaVerbosity verbosity;
extern std::array<int, 4> dim;
extern int &xdim;
extern int &ydim;
extern int &zdim;
extern int &tdim;
