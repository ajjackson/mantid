// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MantidDataObjects/CoordTransformDistance.h"
#include "MantidAPI/CoordTransform.h"
#include "MantidKernel/Exception.h"

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

using namespace Mantid::Geometry;
using namespace Mantid::Kernel;
using Mantid::API::CoordTransform;

namespace Mantid::DataObjects {

//----------------------------------------------------------------------------------------------
/** Constructor
 *
 * @param inD :: # of input dimensions
 * @param center :: array of size[inD], with the coordinates at the center
 * @param dimensionsUsed :: bool array of size[inD] where True is set for those
 *dimensions that are considered when
 *        calculating distance.
 * @param outD :: # of output dimensions
 * @param eigenvects :: eigenvectors of an ellipsoid (if used)
 * @param eigenvals :: variances along each eigenvector (if used)
 */
CoordTransformDistance::CoordTransformDistance(const size_t inD, const coord_t *center, const bool *dimensionsUsed,
                                               const size_t outD, const std::vector<Kernel::V3D> &eigenvects,
                                               const std::vector<double> &eigenvals)
    : CoordTransform(inD, outD) {

  m_center.reserve(inD);
  m_dimensionsUsed.reserve(inD);
  m_eigenvals.reserve(inD);
  m_maxEigenval = 0.0;
  for (size_t d = 0; d < inD; d++) {
    m_center.push_back(center[d]);
    m_dimensionsUsed.push_back(dimensionsUsed[d]);
    if (eigenvals.size() == inD && eigenvects.size() == inD) {
      // coord transform for n-ellipsoid specified
      m_eigenvals.push_back(eigenvals[d]);
      m_eigenvects.push_back(eigenvects[d]);
      m_maxEigenval = std::max(m_maxEigenval, eigenvals[d]);
    }
  }
}

//----------------------------------------------------------------------------------------------
/** Virtual cloner
 * @return a copy of this object  */
CoordTransform *CoordTransformDistance::clone() const { return new CoordTransformDistance(*this); }

//----------------------------------------------------------------------------------------------
/** Apply the coordinate transformation.
 *
 * Calculate the SQUARE of the distance between the input coordinates to
 *m_center
 * but only on dimensionsUsed[d] == true.
 *
 * @param inputVector :: fixed-size array of input coordinates, of size inD
 * @param outVector :: fixed-size array of output coordinates, of size 1
 */
void CoordTransformDistance::apply(const coord_t *inputVector, coord_t *outVector) const {
  if (outD == 1) {
    coord_t distanceSquared = 0;
    if (m_eigenvals.size() == 3) {
      // ellipsoid transfrom with ratio of axes given by standard deviation
      // i.e. sqrt(eigenvals)
      for (size_t d = 0; d < inD; d++) {
        // do dot prod with eigenvector
        coord_t dist = 0.0;
        // calculate the covariance of the delta vector using
        // the three eigen vector as the new base
        for (size_t dd = 0; dd < inD; dd++) {
          dist += static_cast<coord_t>(m_eigenvects[d][dd]) * (inputVector[dd] - m_center[dd]);
        }

        distanceSquared += (dist * dist) * static_cast<coord_t>(m_maxEigenval / m_eigenvals[d]);
      }
      // debug output
      // std::cout << "[CoordTransFormDistance]\n"
      //           << "eigenvals:" << m_eigenvals[0] << "," << m_eigenvals[1]
      //           << "," << m_eigenvals[2] << "\n"
      //           << "peak_center:" << m_center[0] << "," << m_center[1] << ","
      //           << m_center[2] << "\n"
      //           << "distanceSquared = " << distanceSquared << "\n\n";
    } else {
      // nd spherical transform
      for (size_t d = 0; d < inD; d++) {
        if (m_dimensionsUsed[d]) {
          coord_t dist = inputVector[d] - m_center[d];
          distanceSquared += (dist * dist);
        }
      }
    }
    /// Return the only output dimension
    outVector[0] = distanceSquared;
  } else {
    // Cylinder 2D output radius and length
    coord_t lenQdata = 0.0;
    coord_t lenQpeak = 0.0;
    coord_t cosAng = 0.0;
    for (size_t d = 0; d < inD; d++) {
      lenQdata += inputVector[d] * inputVector[d];
      lenQpeak += m_center[d] * m_center[d];
      cosAng += m_center[d] * inputVector[d];
    }
    lenQdata = std::sqrt(lenQdata);
    lenQpeak = std::sqrt(lenQpeak);
    if (lenQpeak * lenQdata != 0.0)
      cosAng /= (lenQpeak * lenQdata);
    coord_t angle = std::acos(cosAng);
    outVector[0] = lenQdata * std::sin(angle);
    outVector[1] = lenQdata * cosAng - lenQpeak;
  }
}

//----------------------------------------------------------------------------------------------
/** Serialize the coordinate transform distance
 *
 * @return The coordinate transform distance in its serialized form.
 */
std::string CoordTransformDistance::toXMLString() const {
  using namespace Poco::XML;

  AutoPtr<Document> pDoc = new Document;
  AutoPtr<Element> coordTransformElement = pDoc->createElement("CoordTransform");
  pDoc->appendChild(coordTransformElement);

  AutoPtr<Element> coordTransformTypeElement = pDoc->createElement("Type");
  coordTransformTypeElement->appendChild(AutoPtr<Text>(pDoc->createTextNode("CoordTransformDistance")));
  coordTransformElement->appendChild(coordTransformTypeElement);

  AutoPtr<Element> paramListElement = pDoc->createElement("ParameterList");

  AutoPtr<Text> formatText = pDoc->createTextNode("%s%s%s%s");
  paramListElement->appendChild(formatText);
  coordTransformElement->appendChild(paramListElement);

  std::stringstream xmlstream;

  DOMWriter writer;
  writer.writeNode(xmlstream, pDoc);

  // Convert the members to parameters
  Mantid::API::InDimParameter inD_param(inD);
  Mantid::API::OutDimParameter outD_param(outD);
  CoordCenterVectorParam m_center_param(inD);
  DimensionsUsedVectorParam m_dimensionsUsed_param(inD);

  // Create and copy the arrays.
  for (size_t d = 0; d < inD; d++) {
    m_center_param.addValue(d, m_center[d]);
    m_dimensionsUsed_param.addValue(d, m_dimensionsUsed[d]);
  }

  std::string formattedXMLString = boost::str(boost::format(xmlstream.str().c_str()) % inD_param.toXMLString().c_str() %
                                              outD_param.toXMLString().c_str() % m_center_param.toXMLString().c_str() %
                                              m_dimensionsUsed_param.toXMLString().c_str());

  return formattedXMLString;
}

/**
 * Coordinate transform id
 * @return the type of coordinate transform
 */
std::string CoordTransformDistance::id() const { return "CoordTransformDistance"; }

} // namespace Mantid::DataObjects
