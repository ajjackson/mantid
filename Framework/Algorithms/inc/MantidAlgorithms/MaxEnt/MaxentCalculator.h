#ifndef MANTID_ALGORITHMS_MAXENTCALCULATOR_H_
#define MANTID_ALGORITHMS_MAXENTCALCULATOR_H_

#include "MantidAlgorithms/DllConfig.h"
#include "MantidAlgorithms/MaxEnt/MaxentCoefficients.h"
#include "MantidAlgorithms/MaxEnt/MaxentEntropy.h"
#include "MantidAlgorithms/MaxEnt/MaxentTransform.h"
#include <boost/shared_ptr.hpp>

namespace Mantid {
namespace Algorithms {

/** MaxentCalculator : TODO: Search directions and
  quadratic coefficients are calculated following J. Skilling and R. K. Bryan:
  "Maximum entropy image reconstruction: general algorithm" (1984), section 3.6.

  Copyright &copy; 2016 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
  National Laboratory & European Spallation Source

  This file is part of Mantid.

  Mantid is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  Mantid is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.

  File change history is stored at: <https://github.com/mantidproject/mantid>
  Code Documentation is available at: <http://doxygen.mantidproject.org>
*/

class MANTID_ALGORITHMS_DLL MaxentCalculator final {
public:
	// Constructor
  MaxentCalculator(MaxentEntropy_sptr entropy, MaxentTransform_sptr transform);
	// Deleted default constructor
  MaxentCalculator() = delete;
	// Destructor
  virtual ~MaxentCalculator() = default;

  // Calculates quadratic coefficients
  void calculateQuadraticCoefficients(const std::vector<double> &data,
                                      const std::vector<double> &errors,
                                      const std::vector<double> &image,
                                      double background);

	// Returns the reconstructed (calculated) data
  std::vector<double> getReconstructedData() const;
  // Returns the image
  std::vector<double> getImage() const;
  // Returns the quadratic coefficients
  QuadraticCoefficients getQuadraticCoefficients() const;
  // Returns the angle between Grad(S) and Grad(C)
  double getAngle() const;
  // Returns the chi-square
  double getChisq();

	// Updates the image
	void updateImage(const std::vector<double> &delta);

private:
  // Calculates the gradient of chi
  std::vector<double> calculateChiGrad() const;
  // Calculates the entropy
  std::vector<double> calculateEntropy() const;
  // Returns the search directions
  std::vector<std::vector<double>> getSearchDirections() const;
  // Corrects the image
  void correctImage();
  // Calculates chi-square
  void calculateChisq();

  // Member variables
  // The experimental (measured) data
  std::vector<double> m_data;
  // The experimental (measured) errors
  std::vector<double> m_errors;
  // The image
  std::vector<double> m_image;
  // The reconstructed (calculated) data
  std::vector<double> m_dataCalc;
  // The background
  double m_background;
  // The angle between Grad(C) and Grad(S)
  double m_angle;
  // Chi-square
  double m_chisq;
  // The type of entropy
  MaxentEntropy_sptr m_entropy;
  // The type of transform
  MaxentTransform_sptr m_transform;
  // The search directions
  std::vector<std::vector<double>> m_directionsIm;
  // The quadratic coefficients
  QuadraticCoefficients m_coeffs;
};

// Helper typedef for scoped pointer of this type.
typedef boost::shared_ptr<MaxentCalculator> MaxentCalculator_sptr;

} // namespace Algorithms
} // namespace Mantid

#endif /* MANTID_ALGORITHMS_MAXENTCALCULATOR_H_ */