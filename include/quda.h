#pragma once

/**
 * @file  quda.h
 * @brief Main header file for the QUDA library
 *
 * Note to QUDA developers: When adding new members to QudaGaugeParam
 * and QudaInvertParam, be sure to update lib/check_params.h as well
 * as the Fortran interface in lib/quda_fortran.F90.
 */

#include <enum_quda.h>
#include <stdio.h> /* for FILE */
#include <quda_define.h>
#include <quda_constants.h>

#ifndef __CUDACC_RTC__
#define double_complex double _Complex
#else // keep NVRTC happy since it can't handle C types
#define double_complex double2
#endif

#ifdef __cplusplus
extern "C" {
#endif

  /**
   * First test function in DSA
   * 
   * Perform a vector scaling
   * 
   * @param[in]  sf The scaling factor
   * @param[in]  data The data to be scaled
   * @param[in]  N The number of elements.
   * @param[in]  prec The precision of the input elements.
   */
  void vector_scale(double sf, void *data, unsigned long long int N, QudaPrecision prec);

  /**
   * Second test function in DSA
   * 
   * Perform a vector reduction (FIXME: add transform options (max, min, etc)
   * 
   * @param[in]  data The data to be reduced
   * @param[in]  N The number of elements.
   * @param[in]  prec The precision of the input elements.
   * @param[ret] The reduction result of the elements in the array.
   */
  double vector_reduce(void *data, unsigned long long int N, QudaPrecision prec);
  
  typedef struct QudaBLASParam_s {
    size_t struct_size; /**< Size of this struct in bytes.  Used to ensure that the host application and QUDA see the same struct*/
    
    QudaBLASType blas_type; /**< Type of BLAS computation to perfrom */

    // GEMM params
    QudaBLASOperation trans_a; /**< operation op(A) that is non- or (conj.) transpose. */
    QudaBLASOperation trans_b; /**< operation op(B) that is non- or (conj.) transpose. */
    int m;                     /**< number of rows of matrix op(A) and C. */
    int n;                     /**< number of columns of matrix op(B) and C. */
    int k;                     /**< number of columns of op(A) and rows of op(B). */
    int lda;                   /**< leading dimension of two-dimensional array used to store the matrix A. */
    int ldb;                   /**< leading dimension of two-dimensional array used to store matrix B. */
    int ldc;                   /**< leading dimension of two-dimensional array used to store matrix C. */
    int a_offset;              /**< position of the A array from which begin read/write. */
    int b_offset;              /**< position of the B array from which begin read/write. */
    int c_offset;              /**< position of the C array from which begin read/write. */
    int a_stride;              /**< stride of the A array in strided(batched) mode */
    int b_stride;              /**< stride of the B array in strided(batched) mode */
    int c_stride;              /**< stride of the C array in strided(batched) mode */
    double_complex alpha; /**< scalar used for multiplication. */
    double_complex beta;  /**< scalar used for multiplication. If beta==0, C does not have to be a valid input. */

    // LU inversion params
    int inv_mat_size; /**< The rank of the square matrix in the LU inversion */

    // Common params
    int batch_count;              /**< number of pointers contained in arrayA, arrayB and arrayC. */
    QudaBLASDataType data_type;   /**< Specifies if using S(C) or D(Z) BLAS type */
    QudaBLASDataOrder data_order; /**< Specifies if using Row or Column major */

  } QudaBLASParam;
  
  /*
   * Interface functions, found in interface_quda.cpp
   */

  /**
   * Set parameters related to status reporting.
   *
   * In typical usage, this function will be called once (or not at
   * all) just before the call to initQuda(), but it's valid to call
   * it any number of times at any point during execution.  Prior to
   * the first time it's called, the parameters take default values
   * as indicated below.
   *
   * @param verbosity  Default verbosity, ranging from QUDA_SILENT to
   *                   QUDA_DEBUG_VERBOSE.  Within a solver, this
   *                   parameter is overridden by the "verbosity"
   *                   member of QudaInvertParam.  The default value
   *                   is QUDA_SUMMARIZE.
   *
   * @param prefix     String to prepend to all messages from QUDA.  This
   *                   defaults to the empty string (""), but you may
   *                   wish to specify something like "QUDA: " to
   *                   distinguish QUDA's output from that of your
   *                   application.
   *
   * @param outfile    File pointer (such as stdout, stderr, or a handle
   *                   returned by fopen()) where messages should be
   *                   printed.  The default is stdout.
   */
  void setVerbosityQuda(QudaVerbosity verbosity, const char prefix[],
                        FILE *outfile);

  /**
   * initCommsGridQuda() takes an optional "rank_from_coords" argument that
   * should be a pointer to a user-defined function with this prototype.
   *
   * @param coords  Node coordinates
   * @param fdata   Any auxiliary data needed by the function
   * @return        MPI rank or QMP node ID cooresponding to the node coordinates
   *
   * @see initCommsGridQuda
   */
  typedef int (*QudaCommsMap)(const int *coords, void *fdata);

  /**
   * @param mycomm User provided MPI communicator in place of MPI_COMM_WORLD
   */

  void qudaSetCommHandle(void *mycomm);

  /**
   * Declare the grid mapping ("logical topology" in QMP parlance)
   * used for communications in a multi-GPU grid.  This function
   * should be called prior to initQuda().  The only case in which
   * it's optional is when QMP is used for communication and the
   * logical topology has already been declared by the application.
   *
   * @param nDim   Number of grid dimensions.  "4" is the only supported
   *               value currently.
   *
   * @param dims   Array of grid dimensions.  dims[0]*dims[1]*dims[2]*dims[3]
   *               must equal the total number of MPI ranks or QMP nodes.
   *
   * @param func   Pointer to a user-supplied function that maps coordinates
   *               in the communication grid to MPI ranks (or QMP node IDs).
   *               If the pointer is NULL, the default mapping depends on
   *               whether QMP or MPI is being used for communication.  With
   *               QMP, the existing logical topology is used if it's been
   *               declared.  With MPI or as a fallback with QMP, the default
   *               ordering is lexicographical with the fourth ("t") index
   *               varying fastest.
   *
   * @param fdata  Pointer to any data required by "func" (may be NULL)
   *
   * @see QudaCommsMap
   */

  void initCommsGridQuda(int nDim, const int *dims, QudaCommsMap func, void *fdata);

  /**
   * Initialize the library.  This is a low-level interface that is
   * called by initQuda.  Calling initQudaDevice requires that the
   * user also call initQudaMemory before using QUDA.
   *
   * @param device CUDA device number to use.  In a multi-GPU build,
   *               this parameter may either be set explicitly on a
   *               per-process basis or set to -1 to enable a default
   *               allocation of devices to processes.
   */
  void initQudaDevice(int device);

  /**
   * Initialize the library persistant memory allocations (both host
   * and device).  This is a low-level interface that is called by
   * initQuda.  Calling initQudaMemory requires that the user has
   * previously called initQudaDevice.
   */
  void initQudaMemory();

  /**
   * Initialize the library.  This function is actually a wrapper
   * around calls to initQudaDevice() and initQudaMemory().
   *
   * @param device  CUDA device number to use.  In a multi-GPU build,
   *                this parameter may either be set explicitly on a
   *                per-process basis or set to -1 to enable a default
   *                allocation of devices to processes.
   */
  void initQuda(int device);

  /**
   * Finalize the library.
   */
  void endQuda(void);

  
  /**
   * A new QudaBLASParam should always be initialized immediately
   * after it's defined (and prior to explicitly setting its members)
   * using this function.  Typical usage is as follows:
   *
   *   QudaBLASParam blas_param = newQudaBLASParam();
   */
  QudaBLASParam newQudaBLASParam(void);
  
  /**
   * Print the members of QudaBLASParam.
   * @param param The QudaBLASParam whose elements we are to print.
   */
  void printQudaBLASParam(QudaBLASParam *param);

  /**
   * @brief Strided Batched GEMM
   * @param[in] arrayA The array containing the A matrix data
   * @param[in] arrayB The array containing the B matrix data
   * @param[in] arrayC The array containing the C matrix data
   * @param[in] native Boolean to use either the native or generic version
   * @param[in] param The data defining the problem execution.
   */
  void blasGEMMQuda(void *arrayA, void *arrayB, void *arrayC, QudaBoolean native, QudaBLASParam *param);

  /**
   * @brief Strided Batched in-place matrix inversion via LU
   * @param[in] Ainv The array containing the A inverse matrix data
   * @param[in] A The array containing the A matrix data
   * @param[in] use_native Boolean to use either the native or generic version
   * @param[in] param The data defining the problem execution.
   */
  void blasLUInvQuda(void *Ainv, void *A, QudaBoolean use_native, QudaBLASParam *param);
  
  void setMPICommHandleQuda(void *mycomm);
  
#ifdef __cplusplus
}
#endif

// remove NVRTC WAR
#undef double_complex

/* #include <quda_new_interface.h> */
