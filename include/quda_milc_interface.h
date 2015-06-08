#ifndef _QUDA_MILC_INTERFACE_H
#define _QUDA_MILC_INTERFACE_H

#include <enum_quda.h>
#include <quda.h>

/**
 * @file    quda_milc_interface.h
 *
 * @section Description
 *
 * The header file defines the milc interface to enable easy
 * interfacing between QUDA and the MILC software packed.
 */

#ifdef __cplusplus
extern "C" {
#endif

  /**
   * Parameters related to linear solvers. 
   */
  typedef struct {
    int max_iter; /** Maximum number of iterations */
    QudaParity evenodd; /** Which parity are we working on ? (options are QUDA_EVEN_PARITY, QUDA_ODD_PARITY, QUDA_INVALID_PARITY */
    int mixed_precision; /** Whether to use mixed precision or not (1 - yes, 0 - no) */
    double boundary_phase[4]; /** Boundary conditions */
  } QudaInvertArgs_t;


  /**
   * Parameters related to problem size and machine topology. 
   */
  typedef struct {
    const int* latsize; /** Local lattice dimensions */
    const int* machsize; /** Machine grid size */
    int device; /** GPU device  number */
  } QudaLayout_t; 


  /**
   * Parameters used to create a QUDA context.
   */
  typedef struct {
    QudaVerbosity verbosity; /** How verbose QUDA should be (QUDA_SILENT, QUDA_VERBOSE or QUDA_SUMMARIZE) */
    QudaLayout_t layout; /** Layout for QUDA to use */
  } QudaInitArgs_t; // passed to the initialization struct


  /**
   * Parameters for defining HISQ calculations
   */
  typedef struct {
    int reunit_allow_svd;         /** Allow SVD for reuniarization */
    int reunit_svd_only;          /** Force use of SVD for reunitarization */
    double reunit_svd_abs_error;  /** Absolute error bound for SVD to apply */
    double reunit_svd_rel_error;  /** Relative error bound for SVD to apply */
    double force_filter;          /** UV filter to apply to force */
  } QudaHisqParams_t;


  /**
   * Parameters for defining fat-link calculations
   */
  typedef struct {
    int su3_source;          /** is the incoming gauge field SU(3) */
    int use_pinned_memory;   /** use page-locked memory in QUDA    */
  } QudaFatLinkArgs_t;

  /**
   * Initialize the QUDA context.
   * 
   * @param input Meta data for the QUDA context
   */
  void qudaInit(QudaInitArgs_t input);

  /**
   * Set set the local dimensions and machine topology for QUDA to use
   *
   * @param layout Struct defining local dimensions and machine topology
   */
  void qudaSetLayout(QudaLayout_t layout);

  /**
   * Destroy the QUDA context.
   */
  void qudaFinalize();

  /**
   * Set the algorithms to use for HISQ fermion calculations, e.g.,
   * SVD parameters for reunitarization.
   *
   * @param hisq_params Meta data desribing the algorithms to use for HISQ fermions
   */
  void qudaHisqParamsInit(QudaHisqParams_t hisq_params);

  /**
   * Compute the fat and long links using the input gauge field.  All
   * fields passed here are host fields, that must be preallocated.
   * The precision of all fields must match.
   *
   * @param precision The precision of the fields
   * @param fatlink_args Meta data for the algorithms to deploy
   * @param act_path_coeff Array of coefficients for each path in the action
   * @param inlink Host gauge field used for input
   * @param fatlink Host fat-link field that is computed
   * @param longlink Host long-link field that is computed
   */ 
  void qudaLoadKSLink(int precision,
		      QudaFatLinkArgs_t fatlink_args,
		      const double act_path_coeff[6],
		      void* inlink,
		      void* fatlink,
		      void* longlink);

  /**
   * Compute the fat links and unitzarize using the input gauge field.
   * All fields passed here are host fields, that must be
   * preallocated.  The precision of all fields must match.
   *
   * @param precision The precision of the fields
   * @param fatlink_args Meta data for the algorithms to deploy
   * @param path_coeff Array of coefficients for each path in the action
   * @param inlink Host gauge field used for input
   * @param fatlink Host fat-link field that is computed
   * @param ulink Host unitarized field that is computed
   */ 
  void qudaLoadUnitarizedLink(int precision,
			      QudaFatLinkArgs_t fatlink_args,
			      const double path_coeff[6],
			      void* inlink,
			      void* fatlink,
			      void* ulink);


  /**
   * Solve Ax=b using an improved staggered operator with a
   * domain-decomposition preconditioner.  All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function requires that persistent gauge and clover fields have
   * been created prior.  This interface is experimental.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param mass Fermion mass parameter
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Target residual
   * @param target_relative_residual Target Fermilab residual
   * @param domain_overlap Array specifying the overlap of the domains in each dimension
   * @param fatlink Fat-link field on the host
   * @param longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual
   * @param final_relative_residual True Fermilab residual
   * @param num_iters Number of iterations taken
   */
  void qudaDDInvert(int external_precision,
		    int quda_precision,
		    double mass,
		    QudaInvertArgs_t inv_args,
		    double target_residual,
		    double target_fermilab_residual,
		    const int * const domain_overlap,
		    const void* const fatlink,
		    const void* const longlink,
		    void* source,
		    void* solution,
		    double* const final_residual,
		    double* const final_fermilab_residual,
		    int* num_iters);

  /**
   * Solve Ax=b using an improved staggered operator with a
   * domain-decomposition preconditioner.  All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function requires that persistent gauge and clover fields have
   * been created prior.  This interface is experimental.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param mass Fermion mass parameter
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Target residual
   * @param target_relative_residual Target Fermilab residual
   * @param domain_overlap Array specifying the overlap of the domains in each dimension
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param tadpole Tadpole improvement facter
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual
   * @param final_relative_residual True Fermilab residual
   * @param num_iters Number of iterations taken
   */
  void qudaInvert(int external_precision,
		  int quda_precision,
		  double mass,
		  QudaInvertArgs_t inv_args,
		  double target_residual,
		  double target_fermilab_residual,
		  const void* const milc_fatlink,
		  const void* const milc_longlink,
		  const double tadpole,
		  void* source,
		  void* solution,
		  double* const final_resid,
		  double* const final_rel_resid,
		  int* num_iters); 
  
 /**
   * Solve  using an improved
   * staggered operator with a domain-decomposition preconditioner.
   * All fields are fields passed and returned are host (CPU) field in
   * MILC order.  This function requires that persistent gauge and
   * clover fields have been created prior.  When a pure
   * double-precision solver is requested no reliable updates are
   * used, else reliable updates are used with a reliable_delta
   * parameter of 0.1.  This interface is experimental.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Target residual
   * @param target_relative_residual Target Fermilab residual
   * @param domain_overlap Array specifying the overlap of the domains in each dimension
   * @param fatlink Fat-link field on the host
   * @param longlink Long-link field on the host
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual
   * @param final_relative_residual True Fermilab residual
   * @param num_iters Number of iterations taken
   */
  void qudaDDInvert(int external_precision,
		    int quda_precision,
		    double mass,
		    QudaInvertArgs_t inv_args,
		    double target_residual,
		    double target_fermilab_residual,
		    const int * const domain_overlap,
		    const void* const fatlink,
		    const void* const longlink,
		    void* source,
		    void* solution,
		    double* const final_residual,
		    double* const final_fermilab_residual,
		    int* num_iters);

  /**
   * Solve for multiple shifts (e.g., masses) using an improved
   * staggered operator.  All fields are fields passed and returned
   * are host (CPU) field in MILC order.  This function requires that
   * persistent gauge and clover fields have been created prior.  When
   * a pure double-precision solver is requested no reliable updates
   * are used, else reliable updates are used with a reliable_delta
   * parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Array of target residuals per shift
   * @param target_relative_residual Array of target Fermilab residuals per shift
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param tadpole Tadpole improvement factor
   * @param source Right-hand side source field
   * @param solutionArray Array of solution spinor fields
   * @param final_residual Array of true residuals
   * @param final_relative_residual Array of true Fermilab residuals
   * @param num_iters Number of iterations taken
   */
  void qudaMultishiftInvert(
      int external_precision,    
      int precision, 
      int num_offsets,
      double* const offset,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const double* target_fermilab_residual,
      const void* const milc_fatlink,
      const void* const milc_longlink,
      const double tadpole,
      void* source,
      void** solutionArray, 
      double* const final_residual,
      double* const final_fermilab_residual,
      int* num_iters);

 /**
   * Solve for a system with many RHS using an improved
   * staggered operator.  
   * The solving procedure consists of two computation phases : 
   * 1) incremental pahse : call eigCG solver to accumulate low eigenmodes
   * 2) deflation phase : use computed eigenmodes to deflate a regular CG
   * All fields are fields passed and returned
   * are host (CPU) field in MILC order.  This function requires that
   * persistent gauge and clover fields have been created prior.  
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Array of target residuals per shift
   * @param target_relative_residual Array of target Fermilab residuals per shift
   * @param milc_fatlink Fat-link field on the host
   * @param milc_longlink Long-link field on the host
   * @param tadpole Tadpole improvement factor
   * @param source Right-hand side source field
   * @param solution Array of solution spinor fields
   * @param ritzVects Array of ritz vectors (may be input or output, depending on a computation phase)
   * @param ritzVals Array of ritz values (may be input or output, depending on a computation phase)
   * @param ritz_prec Precision of the ritz vectors (2 - double, 1 - single)
   * @param max_search_dim eigCG parameter: search space dimention
   * @param nev eigCG parameter: how many eigenpairs to compute within one eigCG call
   * @param deflation_grid eigCG parameter : how many eigenpairs to compute within the incremental phase (# of eigenpairs = nev*deflation_grid)
   * @param tol_restart initCG parameter : at what tolerance value to restart initCG solver 
   * @param rhs_idx  bookkeep current rhs
   * @param last_rhs_flag  is this the last rhs to solve?
   * @param final_residual Array of true residuals
   * @param final_relative_residual Array of true Fermilab residuals
   * @param num_iters Number of iterations taken
   */

  void qudaEigCGInvert(
      int external_precision, 
      int quda_precision,
      double mass,
      QudaInvertArgs_t inv_args,
      double target_residual, 
      double target_fermilab_residual,
      const void* const fatlink,
      const void* const longlink,
      const double tadpole,
      void* source,
      void* solution,
      void* ritzVects,//array of ritz vectors
      double* ritzVals,//array of ritz values
      int ritz_prec,
      const int max_search_dim,
      const int nev,
      const int deflation_grid,
      double tol_restart,//e.g.: 5e+3*target_residual
      const int rhs_idx,//current rhs
      const int last_rhs_flag,//is this the last rhs to solve?
      double* const final_residual,
      double* const final_fermilab_residual,
      int *num_iters);


  /**
   * Solve Ax=b using a Wilson-Clover operator.  All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function creates the gauge and clover field from the host fields.
   * Reliable updates are used with a reliable_delta parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Target residual
   * @param milc_link Gauge field on the host
   * @param milc_clover Clover field on the host
   * @param milc_clover_inv Inverse clover on the host
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param final_residual True residual returned by the solver
   * @param final_residual True Fermilab residual returned by the solver
   * @param num_iters Number of iterations taken
   */
  void qudaCloverInvert(int external_precision, 
			int quda_precision,
			double kappa,
			double clover_coeff,
			QudaInvertArgs_t inv_args,
			double target_residual,
			double target_fermilab_residual,
			const void* milc_link,
			void* milc_clover, 
			void* milc_clover_inv,
			void* source,
			void* solution,
			double* const final_residual, 
			double* const final_fermilab_residual,
			int* num_iters);

  /**
   * Solve for a system with many RHS using using a Wilson-Clover operator.  
   * The solving procedure consists of two computation phases : 
   * 1) incremental pahse : call eigCG solver to accumulate low eigenmodes
   * 2) deflation phase : use computed eigenmodes to deflate a regular CG
   * All fields are fields passed and returned
   * are host (CPU) field in MILC order.  This function requires that
   * persistent gauge and clover fields have been created prior.  
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Target residual
   * @param milc_link Gauge field on the host
   * @param milc_clover Clover field on the host
   * @param milc_clover_inv Inverse clover on the host
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param solution Solution spinor field
   * @param ritzVects Array of ritz vectors (may be input or output, depending on a computation phase)
   * @param ritzVals Array of ritz values (may be input or output, depending on a computation phase)
   * @param ritz_prec Precision of the ritz vectors (2 - double, 1 - single)
   * @param max_search_dim eigCG parameter: search space dimention
   * @param nev eigCG parameter: how many eigenpairs to compute within one eigCG call
   * @param deflation_grid eigCG parameter : how many eigenpairs to compute within the incremental phase (# of eigenpairs = nev*deflation_grid)
   * @param tol_restart initCG parameter : at what tolerance value to restart initCG solver 
   * @param rhs_idx  bookkeep current rhs
   * @param last_rhs_flag  is this the last rhs to solve?
   * @param final_residual Array of true residuals
   * @param final_relative_residual Array of true Fermilab residuals
   * @param num_iters Number of iterations taken
   */


  void qudaEigCGCloverInvert(
      int external_precision, 
      int quda_precision,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      double target_residual, 
      double target_fermilab_residual,
      const void* milc_link,
      void* milc_clover, 
      void* milc_clover_inv,
      void* source,
      void* solution,
      void* ritzVects,//array of ritz vectors
      double* ritzVals,//array of ritz values
      int ritz_prec,
      const int max_search_dim,
      const int nev,
      const int deflation_grid,
      double tol_restart,//e.g.: 5e+3*target_residual
      const int rhs_idx,//current rhs
      const int last_rhs_flag,//is this the last rhs to solve?
      double* const final_residual,
      double* const final_fermilab_residual,
      int *num_iters);

  /**
   * Load the gauge field from the host.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param inv_args Meta data
   * @param milc_link Base pointer to host gauge field (regardless of dimensionality)
   */
  void qudaLoadGaugeField(int external_precision, 
			  int quda_precision,
			  QudaInvertArgs_t inv_args,
			  const void* milc_link) ;

  /**
     Free the gauge field allocated in QUDA.
   */
  void qudaFreeGaugeField();

  /**
   * Load the clover field and its inverse from the host.  If null
   * pointers are passed, the clover field and / or its inverse will
   * be computed dynamically from the resident gauge field.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param inv_args Meta data
   * @param milc_clover Pointer to host clover field.  If 0 then the
   * clover field is computed dynamically within QUDA.
   * @param milc_clover_inv Pointer to host inverse clover field.  If
   * 0 then the inverse if computed dynamically within QUDA.
   * @param solution_type The type of solution required  (mat, matpc)
   * @param solve_type The solve type to use (normal/direct/preconditioning) 
   * @param clover_coeff Clover coefficient
   * @param compute_trlog Whether to compute the trlog of the clover field when inverting
   * @param Array for storing the trlog (length two, one for each parity) 
   */
  void qudaLoadCloverField(int external_precision, 
			   int quda_precision,
			   QudaInvertArgs_t inv_args,
			   void* milc_clover, 
			   void* milc_clover_inv,
			   QudaSolutionType solution_type,
			   QudaSolveType solve_type,
			   double clover_coeff,
			   int compute_trlog,
			   double *trlog) ;

  /**
     Free the clover field allocated in QUDA.
   */
  void qudaFreeCloverField();

  /**
   * Solve for multiple shifts (e.g., masses) using a Wilson-Clover
   * operator with multi-shift CG.  All fields are fields passed and
   * returned are host (CPU) field in MILC order.  This function
   * requires that persistent gauge and clover fields have been
   * created prior.  When a pure double-precision solver is requested
   * no reliable updates are used, else reliable updates are used with
   * a reliable_delta parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Array of target residuals per shift
   * @param milc_link Ignored
   * @param milc_clover Ignored
   * @param milc_clover_inv Ignored
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param solutionArray Array of solution spinor fields
   * @param final_residual Array of true residuals
   * @param num_iters Number of iterations taken
   */
  void qudaCloverMultishiftInvert(int external_precision, 
      int quda_precision,
      int num_offsets,
      double* const offset,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const void* milc_link,
      void* milc_clover, 
      void* milc_clover_inv,
      void* source,
      void** solutionArray,
      double* const final_residual, 
      int* num_iters
      );

  /**
   * Solve for multiple shifts (e.g., masses) using a Wilson-Clover
   * operator with multi-shift CG.  This is a special variant of the
   * multi-shift solver where the additional vectors required for
   * force computation are also returned.  All fields are fields
   * passed and returned are host (CPU) field in MILC order.  This
   * function requires that persistent gauge and clover fields have
   * been created prior.  When a pure double-precision solver is
   * requested no reliable updates are used, else reliable updates are
   * used with a reliable_delta parameter of 0.1.
   *
   * @param external_precision Precision of host fields passed to QUDA (2 - double, 1 - single)
   * @param quda_precision Precision for QUDA to use (2 - double, 1 - single)
   * @param num_offsets Number of shifts to solve for
   * @param offset Array of shift offset values
   * @param kappa Kappa value
   * @param clover_coeff Clover coefficient
   * @param inv_args Struct setting some solver metedata
   * @param target_residual Array of target residuals per shift
   * @param milc_link Ignored
   * @param milc_clover Ignored
   * @param milc_clover_inv Ignored
   * @param clover_coeff Clover coefficient
   * @param source Right-hand side source field
   * @param psiEven Array of solution spinor fields
   * @param psiOdd Array of fields with A_oo^{-1} D_oe * x 
   * @param pEven Array of fields with M_ee * x
   * @param pOdd Array of fields with A_oo^{-1} D_oe * M_ee * x
   * @param final_residual Array of true residuals
   * @param num_iters Number of iterations taken
   */
  void qudaCloverMultishiftMDInvert(int external_precision, 
      int quda_precision,
      int num_offsets,
      double* const offset,
      double kappa,
      double clover_coeff,
      QudaInvertArgs_t inv_args,
      const double* target_residual,
      const void* milc_link,
      void* milc_clover, 
      void* milc_clover_inv,
      void* source,
      void** psiEven,
      void** psiOdd,
      void** pEven,
      void** pOdd,
      double* const final_residual, 
      int* num_iters
      );

  /**
   * Compute the fermion force for the HISQ quark action.  All fields
   * are host fields in MILC order, and the precision of these fields
   * must match.
   *
   * @param precision       The precision of the fields
   * @param level2_coeff    The coefficients for the second level of smearing in the quark action.
   * @param fat7_coeff      The coefficients for the first level of smearing (fat7) in the quark action.
   * @param staple_src      Quark outer-product for the staple.
   * @param one_link_src    Quark outer-product for the one-link term in the action.
   * @param naik_src        Quark outer-product for the three-hop term in the action.
   * @param w_link          Unitarized link variables obtained by applying fat7 smearing and unitarization to the original links.
   * @param v_link          Fat7 link variables. 
   * @param u_link          SU(3) think link variables. 
   * @param milc_momentum        The momentum contribution from the quark action.
   */
  void qudaHisqForce(int precision,
		     const double level2_coeff[6],
		     const double fat7_coeff[6],
		     const void* const staple_src[4],
		     const void* const one_link_src[4],
		     const void* const naik_src[4],
		     const void* const w_link,
		     const void* const v_link,
		     const void* const u_link,
		     void* const milc_momentum);


  /**
   * Compute the fermion force for the Asqtad quark action.  All fields
   * are host fields in MILC order, and the precision of these fields
   * must match.
   *
   * @param precision       The precision of the fields
   * @param act_path_coeff    The coefficients that define the asqtad action.
   * @param one_link_src    Quark outer-product for the one-link term in the action.
   * @param naik_src        Quark outer-product for the three-hop term in the action.
   * @param link            The gauge field
   * @param milc_momentum   The momentum contribution from the quark action.
   */
  void qudaAsqtadForce(int precision,
		       const double act_path_coeff[6],
		       const void* const one_link_src[4],
		       const void* const naik_src[4],
		       const void* const link,
		       void* const milc_momentum);


  /**
   * Compute the gauge force and update the mometum field.  All fields
   * here are CPU fields in MILC order, and their precisions should
   * match.
   *
   * @param precision The precision of the field (2 - double, 1 - single)
   * @param dummy Not presently used
   * @param milc_loop_coeff Coefficients of the different loops in the Symanzik action
   * @param eb3 The integration step size (for MILC this is dt*beta/3)
   * @param milc_sitelink The gauge field from which we compute the force
   * @param milc_momentum The momentum field to be updated
   */
  void qudaGaugeForce(int precision,
		      int dummy,
		      double milc_loop_coeff[3],
		      double eb3,
		      void* milc_sitelink,
		      void* milc_momentum);

  /**
   * Compute the staggered quark-field outer product needed for gauge generation
   *  
   * @param precision The precision of the field (2 - double, 1 - single)
   * @param num_terms The number of quak fields
   * @param coeff The coefficient multiplying the fermion fields in the outer product
   * @param quark_field The input fermion field.
   * @param oprod The outer product to be computed.
   */
  void qudaComputeOprod(int precision,
			int num_terms,
			double** coeff,
			void** quark_field,
			void* oprod[2]);


  /**
   * Evolve the gauge field by step size dt, using the momentum field
   * I.e., Evalulate U(t+dt) = e(dt pi) U(t).  All fields are CPU fields in MILC order.
   *
   * @param precision Precision of the field (2 - double, 1 - single)
   * @param dt The integration step size step
   * @param momentum The momentum field
   * @param The gauge field to be updated 
   */
  void qudaUpdateU(int precision, 
		   double eps,
		   void* momentum, 
		   void* link);
  
  /**
   * Compute the sigma trace field (part of clover force computation).
   * All the pointers here are for QUDA native device objects.  The
   * precisions of all fields must match.  This function requires that
   * there is a persistent clover field.
   * 
   * @param out Sigma trace field  (QUDA device field, geometry = 1)
   * @param dummy (not used)
   * @param mu mu direction
   * @param nu nu direction
   */
  void qudaCloverTrace(void* out,
		       void* dummy,
		       int mu,
		       int nu);


  /**
   * Compute the derivative of the clover term (part of clover force
   * computation).  All the pointers here are for QUDA native device
   * objects.  The precisions of all fields must match.
   * 
   * @param out Clover derivative field (QUDA device field, geometry = 1)
   * @param gauge Gauge field (extended QUDA device field, gemoetry = 4)
   * @param oprod Matrix field (outer product) which is multiplied by the derivative
   * @param mu mu direction
   * @param nu nu direction
   * @param coeff Coefficient of the clover derviative (including stepsize and clover coefficient)
   * @param precision Precision of the fields (2 = double, 1 = single)
   * @param parity Parity for which we are computing
   * @param conjugate Whether to make the oprod field anti-hermitian prior to multiplication
   */
  void qudaCloverDerivative(void* out,
			    void* gauge,
			    void* oprod, 
			    int mu,
			    int nu,
			    double coeff,
			    int precision,
			    int parity,
			    int conjugate);


  /**
   * Take a gauge field on the host, load it onto the device and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the fields (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaCreateExtendedGaugeField(void* gauge,
				     int geometry,
				     int precision);

  /**
   * Take the QUDA resident gauge field and extend it.
   * Return a pointer to the extended gauge field object.
   *
   * @param gauge The CPU gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the fields (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaResidentExtendedGaugeField(void* gauge,
				       int geometry,
				       int precision);

  /**
   * Allocate a gauge (matrix) field on the device and optionally download a host gauge field.
   *
   * @param gauge The host gauge field (optional - if set to 0 then the gauge field zeroed)
   * @param geometry The geometry of the matrix field to create (1 - scaler, 4 - vector, 6 - tensor)
   * @param precision The precision of the field to be created (2 - double, 1 - single)
   * @return Pointer to the gauge field (cast as a void*)
   */
  void* qudaCreateGaugeField(void* gauge,
			     int geometry,
			     int precision);

  /**
   * Copy the QUDA gauge (matrix) field on the device to the CPU
   *
   * @param outGauge Pointer to the host gauge field
   * @param inGauge Pointer to the device gauge field (QUDA device field)
   */
  void qudaSaveGaugeField(void* gauge,
			  void* inGauge);

  /**
   * Reinterpret gauge as a pointer to cudaGaugeField and call destructor.
   *
   * @param gauge Gauge field to be freed
   */
  void qudaDestroyGaugeField(void* gauge);

  
#ifdef __cplusplus
}
#endif


#endif // _QUDA_MILC_INTERFACE_H
