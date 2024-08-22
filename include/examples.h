#pragma once

#include <quda_internal.h>

namespace quda
{
  /**
   * Function that wraps the vector scaling kernel.
   *
   * @param[in] data The data to be scaled
   * @param[in] N The number of data elements
   * @param[in] sf The scaling factor
   * @param[in] prec The precision of the data
   */
  void scaleVector(void *data, const unsigned long long int N, const double sf, const QudaPrecision prec);

  /**
   * Function that wraps the vector reduction kernel.
   *
   * @param[in] data The data to be scaled
   * @param[in] N The number of data elements
   * @param[in] prec The precision of the data
   * @param[ret] The reducuced value of the data
   */
  double reduceVector(void *data, const unsigned long long int N, const QudaPrecision prec);
} // namespace quda
