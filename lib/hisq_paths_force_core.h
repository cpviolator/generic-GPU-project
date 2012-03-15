

template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit> 
  __global__ void
  do_middle_link_kernel(const RealA* const oprodEven, const RealA* const oprodOdd,
			const RealA* const QprevEven, const RealA* const QprevOdd,  
			const RealB* const linkEven,  const RealB* const linkOdd,
			int sig, int mu, 
			typename RealTypeId<RealA>::Type coeff,
			RealA* const PmuEven, RealA* const PmuOdd, 
			RealA* const P3Even, RealA* const P3Odd,
			RealA* const QmuEven, RealA* const QmuOdd, 
			RealA* const newOprodEven, RealA* const newOprodOdd) 
{		
  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int new_x[4];
  int new_mem_idx;
  int ad_link_sign=1;
  int ab_link_sign=1;
  int bc_link_sign=1;

  RealB LINK_W[ArrayLength<RealB>::result];
  RealB LINK_X[ArrayLength<RealB>::result];
  RealB LINK_Y[ArrayLength<RealB>::result];


  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result];
  RealA COLOR_MAT_X[ArrayLength<RealA>::result];
  
  /*        A________B
   *   mu   |        |
   *  	   D|        |C
   *	  
   *	  A is the current point (sid)
   *
   */
  
  int point_b, point_c, point_d;
  int ad_link_nbr_idx, ab_link_nbr_idx, bc_link_nbr_idx;
  int mymu;

  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(mu_positive){
    mymu = mu;
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
  }else{
    mymu = OPP_DIR(mu);
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(OPP_DIR(mu), X, new_mem_idx);	
  }
  point_d = (new_mem_idx >> 1);
  if (mu_positive){
    ad_link_nbr_idx = point_d;
    reconstructSign(&ad_link_sign, mymu, new_x);
  }else{
    ad_link_nbr_idx = sid;
    reconstructSign(&ad_link_sign, mymu, x);	
  }

  int mysig; 
  if(sig_positive){
    mysig = sig;
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
  }else{
    mysig = OPP_DIR(sig);
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
  }
  point_c = (new_mem_idx >> 1);
  if (mu_positive){
    bc_link_nbr_idx = point_c;	
    reconstructSign(&bc_link_sign, mymu, new_x);
  }

  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(sig_positive){
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
  }else{
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
  }
  point_b = (new_mem_idx >> 1); 

  if (!mu_positive){
    bc_link_nbr_idx = point_b;
    reconstructSign(&bc_link_sign, mymu, new_x);
  }   

  if(sig_positive){
    ab_link_nbr_idx = sid;
    reconstructSign(&ab_link_sign, mysig, x);	
  }else{	
    ab_link_nbr_idx = point_b;
    reconstructSign(&ab_link_sign, mysig, new_x);
  }
  // now we have ab_link_nbr_idx


  // load the link variable connecting a and b 
  // Store in LINK_W 
  if(sig_positive){
    loadMatrixFromField(linkEven, linkOdd, mysig, ab_link_nbr_idx, LINK_W, oddBit);
  }else{
    loadMatrixFromField(linkEven, linkOdd, mysig, ab_link_nbr_idx, LINK_W, 1-oddBit);
  }
  
  // load the link variable connecting b and c 
  // Store in LINK_X
  if(mu_positive){
    loadMatrixFromField(linkEven, linkOdd, mymu, bc_link_nbr_idx, LINK_X, oddBit);
  }else{ 
    loadMatrixFromField(linkEven, linkOdd, mymu, bc_link_nbr_idx, LINK_X, 1-oddBit);
  }

  
  if(QprevOdd == NULL){
    if(sig_positive){
      loadMatrixFromField(oprodEven, oprodOdd, sig, point_d, COLOR_MAT_Y, 1-oddBit);
    }else{
      loadAdjointMatrixFromField(oprodEven, oprodOdd, OPP_DIR(sig), point_c, COLOR_MAT_Y, oddBit);
    }
  }else{ // QprevOdd != NULL
    loadMatrixFromField(oprodEven, oprodOdd, point_c, COLOR_MAT_Y, oddBit);
  }
  
  
  MATRIX_PRODUCT(COLOR_MAT_W, LINK_X, COLOR_MAT_Y, !mu_positive);
  if(PmuOdd){
    storeMatrixToField(COLOR_MAT_W, point_b, PmuEven, PmuOdd, 1-oddBit);
  }
  MATRIX_PRODUCT(COLOR_MAT_Y, LINK_W, COLOR_MAT_W, sig_positive);
  storeMatrixToField(COLOR_MAT_Y, sid, P3Even, P3Odd, oddBit);
  
  
  if(mu_positive){
    loadMatrixFromField(linkEven, linkOdd, mymu, ad_link_nbr_idx, LINK_Y, 1-oddBit);
  }else{
    loadAdjointMatrixFromField(linkEven, linkOdd, mymu, ad_link_nbr_idx, LINK_Y, oddBit);
  }
  
  
  if(QprevOdd == NULL){
    if(sig_positive){
      MAT_MUL_MAT(COLOR_MAT_W, LINK_Y, COLOR_MAT_Y);
    }
    if(QmuEven){
      ASSIGN_MAT(LINK_Y, COLOR_MAT_X); 
      storeMatrixToField(COLOR_MAT_X, sid, QmuEven, QmuOdd, oddBit);
    }
  }else{ 
    loadMatrixFromField(QprevEven, QprevOdd, point_d, COLOR_MAT_Y, 1-oddBit);
    MAT_MUL_MAT(COLOR_MAT_Y, LINK_Y, COLOR_MAT_X);
    if(QmuEven){
      storeMatrixToField(COLOR_MAT_X, sid, QmuEven, QmuOdd, oddBit);
    }
    if(sig_positive){
      MAT_MUL_MAT(COLOR_MAT_W, COLOR_MAT_X, COLOR_MAT_Y);
    }	
  }
    
  if(sig_positive){
    addMatrixToField(COLOR_MAT_Y, sig, sid, coeff, newOprodEven, newOprodOdd, oddBit);
  }

  return;
}

template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
  __global__ void
  do_side_link_kernel(const RealA* const P3Even, const RealA* const P3Odd,
		      const RealA* const oprodEven, const RealA* const oprodOdd,
		      const RealB* const linkEven,  const RealB* const linkOdd,
		      int sig, int mu, 
		      typename RealTypeId<RealA>::Type coeff, 
		      typename RealTypeId<RealA>::Type accumu_coeff,
		      RealA* const shortPEven, RealA* const shortPOdd,
		      RealA* const newOprodEven, RealA* const newOprodOdd)
{

  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int ad_link_sign = 1;

  RealB LINK_W[ArrayLength<RealB>::result];

  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
  // The compiler probably knows to reorder so that loads are done early on
  loadMatrixFromField(P3Even, P3Odd, sid, COLOR_MAT_Y, oddBit);

  /*      compute the side link contribution to the momentum
   *
   *             sig
   *          A________B
   *           |       |   mu
   *         D |       |C
   *
   *      A is the current point (sid)
   *
   */

  typename RealTypeId<RealA>::Type mycoeff;
  int point_d;
  int ad_link_nbr_idx;
  int mymu;
  int new_mem_idx;

  int new_x[4];
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(mu_positive){
    mymu=mu;
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mymu,X, new_mem_idx);
  }else{
    mymu = OPP_DIR(mu);
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mymu, X, new_mem_idx);
  }
  point_d = (new_mem_idx >> 1);


  if (mu_positive){
    ad_link_nbr_idx = point_d;
    reconstructSign(&ad_link_sign, mymu, new_x);
  }else{
    ad_link_nbr_idx = sid;
    reconstructSign(&ad_link_sign, mymu, x);	
  }


  if(mu_positive){
    loadMatrixFromField(linkEven, linkOdd, mymu, ad_link_nbr_idx, LINK_W, 1-oddBit);
  }else{
    loadMatrixFromField(linkEven, linkOdd, mymu, ad_link_nbr_idx, LINK_W, oddBit);
  }


  // Should all be inside if (shortPOdd)
  if (shortPOdd){
    MATRIX_PRODUCT(COLOR_MAT_W, LINK_W, COLOR_MAT_Y, mu_positive);
    addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit);
  }


  mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;

  if(oprodOdd){
    loadMatrixFromField(oprodEven, oprodOdd, point_d, COLOR_MAT_X, 1-oddBit);
    if(mu_positive){
      MAT_MUL_MAT(COLOR_MAT_Y, COLOR_MAT_X, COLOR_MAT_W);

      // Added by J.F.
      if(!oddBit){ mycoeff = -mycoeff; }
      addMatrixToField(COLOR_MAT_W, mu, point_d, mycoeff, newOprodEven, newOprodOdd, 1-oddBit);
    }else{
      ADJ_MAT_MUL_ADJ_MAT(COLOR_MAT_X, COLOR_MAT_Y, COLOR_MAT_W);
      if(oddBit){ mycoeff = -mycoeff; }
      addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, newOprodEven, newOprodOdd, oddBit);
    } 
  }

  if(!oprodOdd){
    if(mu_positive){
      if(!oddBit){ mycoeff = -mycoeff;}
      addMatrixToField(COLOR_MAT_Y, mu, point_d, mycoeff, newOprodEven, newOprodOdd, 1-oddBit);
    }else{
      if(oddBit){ mycoeff = -mycoeff; }
      ADJ_MAT(COLOR_MAT_Y, COLOR_MAT_W);
      addMatrixToField(COLOR_MAT_W, OPP_DIR(mu), sid, mycoeff, newOprodEven, newOprodOdd,  oddBit);
    }
  }

  return;
}


template<class RealA, class RealB, int sig_positive, int mu_positive, int oddBit>
  __global__ void
  do_all_link_kernel(const RealA* const oprodEven, const RealA* const oprodOdd, 
		     const RealA* const QprevEven, const RealA* const QprevOdd,
		     const RealB* const linkEven, const RealB* const linkOdd,
		     int sig, int mu, 
		     typename RealTypeId<RealA>::Type coeff, 
		     typename RealTypeId<RealA>::Type accumu_coeff,
		     RealA* const shortPEven, RealA* const shortPOdd,
		     RealA* const newOprodEven, RealA* const newOprodOdd)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;
  int X = 2*sid + x1odd;

  int new_x[4];

  RealB LINK_W[ArrayLength<RealB>::result];
  RealB LINK_X[ArrayLength<RealB>::result];
  RealB LINK_Y[ArrayLength<RealB>::result];
  RealA COLOR_MAT_W[ArrayLength<RealA>::result]; 
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result]; 
  RealA COLOR_MAT_X[ArrayLength<RealA>::result]; 

  /*            sig
   *         A________B
   *      mu  |      |
   *        D |      |C
   *
   *   A is the current point (sid)
   *
   */
  
  int point_b, point_c, point_d;
  int ab_link_nbr_idx, bc_link_nbr_idx;
  int mysig; 
  int new_mem_idx;
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  if(sig_positive){
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, X, new_mem_idx);
  }else{
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), X, new_mem_idx);	
  }
  point_b = (new_mem_idx >> 1);
  ab_link_nbr_idx = (sig_positive) ? sid : point_b;
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];

  
  const typename RealTypeId<RealA>::Type & mycoeff = CoeffSign<sig_positive,oddBit>::result*coeff;
  {
    FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(mu, X, new_mem_idx);
    point_d = (new_mem_idx >> 1);
    loadMatrixFromField(QprevEven, QprevOdd, point_d, COLOR_MAT_X, 1-oddBit);	   // COLOR_MAT_X
    loadMatrixFromField(linkEven, linkOdd, mu, point_d, LINK_Y, 1-oddBit); 	   // LINK_Y used!

    if(sig_positive){
      FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
    }else{
      FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
    }
    point_c = (new_mem_idx >> 1);
    loadMatrixFromField(oprodEven,oprodOdd,  point_c, COLOR_MAT_Y, oddBit);		// COLOR_MAT_Y
    loadMatrixFromField(linkEven, linkOdd, mu, point_c, LINK_W, oddBit);             // LINK_W


    MATRIX_PRODUCT(LINK_X, LINK_W, COLOR_MAT_Y, 0); // COMPUTE_LINK_X
    if (sig_positive)
      {
	MAT_MUL_MAT(COLOR_MAT_X, LINK_Y, COLOR_MAT_Y);
	MAT_MUL_MAT(LINK_X, COLOR_MAT_Y, COLOR_MAT_W);
	addMatrixToField(COLOR_MAT_W, sig, sid, Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, oddBit);
      }

    if (sig_positive){
      loadMatrixFromField(linkEven, linkOdd, sig, ab_link_nbr_idx, LINK_W, oddBit); // LINK_Z used
    }else{
      loadMatrixFromField(linkEven, linkOdd, OPP_DIR(sig), ab_link_nbr_idx, LINK_W, 1-oddBit); // LINK_Z used!
    }
    MATRIX_PRODUCT(COLOR_MAT_Y, LINK_W, LINK_X, sig_positive); // COLOR_MAT_Y is assigned here

    MAT_MUL_MAT(COLOR_MAT_Y, COLOR_MAT_X, COLOR_MAT_W);
    addMatrixToField(COLOR_MAT_W, mu, point_d, -Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, 1-oddBit);

    MAT_MUL_MAT(LINK_Y, COLOR_MAT_Y, COLOR_MAT_W);
    addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit);
  } // positive mu
  {
    new_x[0] = x[0];
    new_x[1] = x[1];
    new_x[2] = x[2];
    new_x[3] = x[3];
    FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(mu, X, new_mem_idx);	
    point_d = (new_mem_idx >> 1);
    loadMatrixFromField(QprevEven, QprevOdd, point_d, COLOR_MAT_X, 1-oddBit);         // COLOR_MAT_X used!
    loadMatrixFromField(linkEven, linkOdd, mu, sid, LINK_Y, oddBit);  	     // LINK_Y used
    
    if(sig_positive){
      FF_COMPUTE_NEW_FULL_IDX_PLUS_UPDATE(sig, new_mem_idx, new_mem_idx);
    }else{
      FF_COMPUTE_NEW_FULL_IDX_MINUS_UPDATE(OPP_DIR(sig), new_mem_idx, new_mem_idx);	
    }
    point_c = (new_mem_idx >> 1);
    loadMatrixFromField(oprodEven, oprodOdd, point_c, COLOR_MAT_Y, oddBit);	     // COLOR_MAT_Y used
    loadMatrixFromField(linkEven, linkOdd, mu, point_b, LINK_W, 1-oddBit);   // LINK_W used 
    
    if(sig_positive){
      MAT_MUL_ADJ_MAT(COLOR_MAT_X, LINK_Y, COLOR_MAT_W);
    }
    MAT_MUL_MAT(LINK_W, COLOR_MAT_Y, LINK_X);
    if (sig_positive){	
	MAT_MUL_MAT(LINK_X, COLOR_MAT_W, COLOR_MAT_Y);
	addMatrixToField(COLOR_MAT_Y, sig, sid, Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, oddBit);
      }

    if (sig_positive){
      loadMatrixFromField(linkEven, linkOdd, sig, ab_link_nbr_idx, LINK_W, oddBit);       // LINK_Z used
    }else{
      loadMatrixFromField(linkEven, linkOdd, OPP_DIR(sig), ab_link_nbr_idx, LINK_W, 1-oddBit); // LINK_Z used!
    }

    MATRIX_PRODUCT(COLOR_MAT_Y, LINK_W, LINK_X, sig_positive);
    ADJ_MAT_MUL_ADJ_MAT(COLOR_MAT_X, COLOR_MAT_Y, COLOR_MAT_W);	
    addMatrixToField(COLOR_MAT_W, mu, sid, Sign<oddBit>::result*mycoeff, newOprodEven, newOprodOdd, oddBit);

    MATRIX_PRODUCT(COLOR_MAT_W, LINK_Y, COLOR_MAT_Y, 0);
    addMatrixToField(COLOR_MAT_W, point_d, accumu_coeff, shortPEven, shortPOdd, 1-oddBit);
  } // negative mu
  return;
}

template<class RealA, int oddBit>
  __global__ void 
  do_one_link_term_kernel(const RealA* const oprodEven, const RealA* const oprodOdd,
			  int sig, typename RealTypeId<RealA>::Type coeff,
			  RealA* const outputEven, RealA* const outputOdd)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  
  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  if(GOES_FORWARDS(sig)){
    loadMatrixFromField(oprodEven, oprodOdd, sig, sid, COLOR_MAT_W, oddBit);
    addMatrixToField(COLOR_MAT_W, sig, sid, coeff, outputEven, outputOdd, oddBit);
  }
  return;
}




template<class RealA, int oddBit>
  __global__ void 
  do_longlink_kernel(const RealA* const linkEven, const RealA* const linkOdd,
		     const RealA* const naikOprodEven, const RealA* const naikOprodOdd,
		     int sig, typename RealTypeId<RealA>::Type coeff,
		     RealA* const outputEven, RealA* const outputOdd)
{
       
  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;

  int new_x[4];
  new_x[0] = x[0];
  new_x[1] = x[1];
  new_x[2] = x[2];
  new_x[3] = x[3];


  RealA LINK_W[ArrayLength<RealA>::result];
  RealA LINK_X[ArrayLength<RealA>::result];
  RealA LINK_Y[ArrayLength<RealA>::result];
  RealA LINK_Z[ArrayLength<RealA>::result];

  RealA COLOR_MAT_U[ArrayLength<RealA>::result];
  RealA COLOR_MAT_V[ArrayLength<RealA>::result];
  RealA COLOR_MAT_W[ArrayLength<RealA>::result]; // used as a temporary
  RealA COLOR_MAT_X[ArrayLength<RealA>::result];
  RealA COLOR_MAT_Y[ArrayLength<RealA>::result];
  RealA COLOR_MAT_Z[ArrayLength<RealA>::result];


  const int & point_c = sid;
  int point_a, point_b, point_d, point_e;
  // need to work these indices
  int X[4];
  X[0] = X1;
  X[1] = X2;
  X[2] = X3;
  X[3] = X4;

  // compute the force for forward long links
  if(GOES_FORWARDS(sig))
    {
      new_x[sig] = (x[sig] + 1 + X[sig])%X[sig];
      point_d = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;
	  
      new_x[sig] = (new_x[sig] + 1 + X[sig])%X[sig];
      point_e = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;
	  
      new_x[sig] = (x[sig] - 1 + X[sig])%X[sig];
      point_b = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;

      new_x[sig] = (new_x[sig] - 1 + X[sig])%X[sig];
      point_a = (new_x[3]*X3X2X1+new_x[2]*X2X1+new_x[1]*X1+new_x[0]) >> 1;

      loadMatrixFromField(linkEven, linkOdd, sig, point_a, LINK_W, oddBit);
      loadMatrixFromField(linkEven, linkOdd, sig, point_b, LINK_X, 1-oddBit);
      loadMatrixFromField(linkEven, linkOdd, sig, point_d, LINK_Y, 1-oddBit);
      loadMatrixFromField(linkEven, linkOdd, sig, point_e, LINK_Z, oddBit);

      loadMatrixFromField(naikOprodEven, naikOprodOdd, sig, point_c, COLOR_MAT_Z, oddBit);
      loadMatrixFromField(naikOprodEven, naikOprodOdd, sig, point_b, COLOR_MAT_Y, 1-oddBit);
      loadMatrixFromField(naikOprodEven, naikOprodOdd, sig, point_a, COLOR_MAT_X, oddBit);

      MAT_MUL_MAT(LINK_Z, COLOR_MAT_Z, COLOR_MAT_W); // link(d)*link(e)*Naik(c)
      MAT_MUL_MAT(LINK_Y, COLOR_MAT_W, COLOR_MAT_V);

      MAT_MUL_MAT(LINK_Y, COLOR_MAT_Y, COLOR_MAT_W);  // link(d)*Naik(b)*link(b)
      MAT_MUL_MAT(COLOR_MAT_W, LINK_X, COLOR_MAT_U);
      SCALAR_MULT_ADD_MATRIX(COLOR_MAT_V, COLOR_MAT_U, -1, COLOR_MAT_V);

      MAT_MUL_MAT(COLOR_MAT_X, LINK_W, COLOR_MAT_W); // Naik(a)*link(a)*link(b)
      MAT_MUL_MAT(COLOR_MAT_W, LINK_X, COLOR_MAT_U);
      SCALAR_MULT_ADD_MATRIX(COLOR_MAT_V, COLOR_MAT_U, 1, COLOR_MAT_V);

      addMatrixToField(COLOR_MAT_V, sig, sid,  coeff, outputEven, outputOdd, oddBit);
    }

  return;
}



template<class RealA, class RealB, int oddBit>
  __global__ void 
  do_complete_force_kernel(const RealB* const linkEven, const RealB* const linkOdd, 
			   const RealA* const oprodEven, const RealA* const oprodOdd,
			   int sig,
			   RealA* const forceEven, RealA* const forceOdd)
{
  int sid = blockIdx.x * blockDim.x + threadIdx.x;

  int x[4];
  int z1 = sid/X1h;
  int x1h = sid - z1*X1h;
  int z2 = z1/X2;
  x[1] = z1 - z2*X2;
  x[3] = z2/X3;
  x[2] = z2 - x[3]*X3;
  int x1odd = (x[1] + x[2] + x[3] + oddBit) & 1;
  x[0] = 2*x1h + x1odd;

  int link_sign;

  RealB LINK_W[ArrayLength<RealB>::result];
  RealA COLOR_MAT_W[ArrayLength<RealA>::result];
  RealA COLOR_MAT_X[ArrayLength<RealA>::result];


  loadMatrixFromField(linkEven, linkOdd, sig, sid, LINK_W, oddBit);
  reconstructSign(&link_sign, sig, x);	

  loadMatrixFromField(oprodEven, oprodOdd, sig, sid, COLOR_MAT_X, oddBit);
	
  typename RealTypeId<RealA>::Type coeff = (oddBit==1) ? -1 : 1;
  MAT_MUL_MAT(LINK_W, COLOR_MAT_X, COLOR_MAT_W);
	
  storeMatrixToMomentumField(COLOR_MAT_W, sig, sid, coeff, forceEven, forceOdd, oddBit); 
  return;
}