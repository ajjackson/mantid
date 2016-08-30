#include "MantidAlgorithms/UnwrapMonitorsInTOF.h"
#include "MantidAPI/WorkspaceProperty.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/ISpectrum.h"
#include "MantidAPI/WorkspaceFactory.h"
#include "MantidGeometry/Instrument.h"
#include "MantidKernel/PhysicalConstants.h"
#include "MantidHistogramData/Histogram.h"
#include "MantidHistogramData/Points.h"
#include "MantidHistogramData/Counts.h"
#include "MantidAPI/ISpectrum.h"
#include <numeric>

namespace
{
  const double specialWavelengthCutoff = -1.0;
  const double specialTimeOfFlightCutoff = -1.0;
  const int specialIndex = -1;



struct MinAndMaxTof {
  MinAndMaxTof(double minTof, double maxTof): minTof(minTof), maxTof(maxTof) {}
  double minTof;
  double maxTof;
};

struct MinAndMaxIndex {
  MinAndMaxIndex(int minIndex, int maxIndex): minIndex(minIndex), maxIndex(maxIndex) {}
  int minIndex;
  int maxIndex;
};

/**
 * For a given distance from the source and lower and upper wavelength bound the function calculates the lower and upper bound of the
 * TOF for this distance.
 *
 * To get the time: T = L/V and V = h/(m*lambda) --> T(lambda) = (L*m/h)*lambda
 * In addition we need to divide the wavelength by 10^10 since the input is in Anstrom and we need to multiply by 10^6 since the
 * output is in microseconds
 * @param distanceFromSource the distance from the source in meters
 * @param lowerWavelengthLimit the lower bound of the wavelength in Angstrom
 * @param upperWavelengthLimit the upper bound of the wavelength in Angstrom
 * @return  the upper and lower bound of the TOF in microseconds
 */
MinAndMaxTof getMinAndMaxTofForDistanceFromSoure(double distanceFromSource, double lowerWavelengthLimit, double upperWavelengthLimit) {
  // Calculate and set the constant factor for the conversion to wavelength
  const double angstromConversion = 1e10;
  const double microsecondConversion = 1e6;
  const double conversionConstant = (distanceFromSource*Mantid::PhysicalConstants::NeutronMass*microsecondConversion)/
                                     (Mantid::PhysicalConstants::h*angstromConversion);
  const double minTof = lowerWavelengthLimit == specialWavelengthCutoff ? specialTimeOfFlightCutoff : conversionConstant*lowerWavelengthLimit;
  const double maxTof = upperWavelengthLimit == specialWavelengthCutoff ? specialTimeOfFlightCutoff : conversionConstant*upperWavelengthLimit;
  return MinAndMaxTof(minTof, maxTof);
}


double getDistanceFromSourceForWorkspaceIndex(Mantid::API::MatrixWorkspace* workspace, size_t workspaceIndex) {
  auto detector = workspace->getDetector(workspaceIndex);
  return detector->getDistance(*(workspace->getInstrument()->getSource()));
}

MinAndMaxTof getMinAndMaxTof(Mantid::API::MatrixWorkspace*workspace, size_t workspaceIndex,
                             double lowerWavelengthLimit, double upperWavelengthLimit) {
  auto distanceFromSource = getDistanceFromSourceForWorkspaceIndex(workspace, workspaceIndex);
  return getMinAndMaxTofForDistanceFromSoure(distanceFromSource, lowerWavelengthLimit, upperWavelengthLimit);
}

/**
 * Doubles up the x points. This means that if the current points are 1, 2, 3, 4 then the new
 * bin edges are 1, 2, 3, 4, 5, 6, 7, 8
 * @param workspace the workspace from which we grab the x data
 * @param workspaceIndex the particular histogram index
 * @return a BinEdges object
 */
Mantid::HistogramData::Points getPoints(Mantid::API::MatrixWorkspace* workspace, size_t workspaceIndex) {
  auto points = workspace->points(workspaceIndex);
  std::vector<double> doubledData(2*points.size(), 0);
  std::copy(std::begin(points), std::end(points), std::begin(doubledData));

  // Calculate the doubled up bit. To do this we need to:
  // 1. Get the last element of the original BinEdges
  // 2. For each entry in the BinEdges calcualte the increment and add it to the last element. In addition
  //    we have to make sure that there is a spacing between the last element and the newly append last+1 element
  //    This spacing is taken to be the difference between the first and the second element
  auto firstTof = points.front();
  auto lastTof = points.back();
  auto doubledDataIterator = doubledData.begin();
  std::advance(doubledDataIterator,points.size());
  double newValue = 0.0;

  double lastElementToNewElementSpacing = 0.0;
  if (doubledData.size() > 1) {
    lastElementToNewElementSpacing = doubledData[1] - doubledData[0];
  }

  for (auto pointsIterator = points.begin(); doubledDataIterator != doubledData.end();
       ++doubledDataIterator, ++pointsIterator) {
    newValue = lastTof + lastElementToNewElementSpacing + (*pointsIterator - firstTof);
    *doubledDataIterator = newValue;
  }

  return Mantid::HistogramData::Points{doubledData};
}

MinAndMaxIndex getMinAndMaxIndex(const MinAndMaxTof& minMaxTof, const Mantid::HistogramData::Points& points) {
  int minIndex = specialIndex;
  int maxIndex = specialIndex;
  const auto minCutOff = minMaxTof.minTof;
  const auto maxCutOff = minMaxTof.maxTof;
  int index = 0;
  for (const auto& element : points) {
    if (element < minCutOff) {
      minIndex = index;
    }

    if (element > maxCutOff) {
        maxIndex = index;
        // We can break here since the min would have been set before the max
        break;
    }
    ++index;
  }

  // We have to take care since the maxIndex can be a special index for two reasons
  // 1. The maxCutOff is smaller than the smallest time of flight value of the workspace, then the special index index is correct
  // 2. The maxCutOff is larger then the largest time of flight value of the workspace, then the last index is correct
  if (maxIndex == specialIndex) {
    if (points[points.size() -1 ] <  maxCutOff) {
        maxIndex = static_cast<int>(points.size()) - 1;
    }
  }

  // The min index is the index which is the largest lower bound index which is not in the TOF region, hence we need index + 1 as the
  // lower bound index
  minIndex += 1;

  return MinAndMaxIndex(minIndex, maxIndex);
}


void setTofBelowLowerBoundToZero(std::vector<double>& doubledData, int minIndex) {
  if (minIndex == specialIndex) {
    return;
  }
  auto begin = doubledData.begin();
  auto end = begin;
  std::advance(end, minIndex);
  std::fill(begin, end, 0.0);
}


void setTofAboveUpperBoundToZero(std::vector<double>& doubledData, int maxIndex) {
  if (maxIndex >= doubledData.size() - 1) {
    return;
  }
  auto begin = doubledData.begin();
  std::advance(begin, maxIndex);
  auto end = doubledData.end();
  std::fill(begin, end, 0.0);
}


/**
 * Creates a doubled up version of the counts for a particular histogram. It sets everything to zero which
 * is not in the valid TOF range
 * @param workspace the workspace from which we grab the valid counts
 * @param workspaceIndex the particular histogram index
 * @param minMaxTof the upper and lower boundary of the TOF.
 * @param binEdges the TOF BinEdges object
 * @return a Counts object
 */
Mantid::HistogramData::Counts getCounts(Mantid::API::MatrixWorkspace* workspace,
                                        size_t workspaceIndex, const MinAndMaxTof& minMaxTof,
                                        const Mantid::HistogramData::Points& points) {
  // Create the data twice
  auto counts = workspace->counts(workspaceIndex);
  std::vector<double> doubledData(2*counts.size(), 0);
  auto doubledDataIterator = doubledData.begin();
  std::copy(std::begin(counts), std::end(counts), doubledDataIterator);
  std::advance(doubledDataIterator,counts.size());
  std::copy(std::begin(counts), std::end(counts), doubledDataIterator);

  // Now set everything to zero entires which correspond to TOF which is less than minTof and more than maxTof
  auto minAndMaxIndex = getMinAndMaxIndex(minMaxTof, points);
  if (minMaxTof.minTof != specialTimeOfFlightCutoff) {
    setTofBelowLowerBoundToZero(doubledData, minAndMaxIndex.minIndex);
  }

  if (minMaxTof.maxTof != specialTimeOfFlightCutoff) {
    setTofAboveUpperBoundToZero(doubledData, minAndMaxIndex.maxIndex);
  }
  return Mantid::HistogramData::Counts(doubledData);
}

/**
 * Get the workspace indices which correspond to the monitors of a workspace. There are three options
 * 1. It is mixed
 * 2. It is purely a monitor workspace
 * 3. There are no moniors
 * @param workspace
 * @return
 */
std::vector<size_t> getWorkspaceIndicesForMonitors(Mantid::API::MatrixWorkspace*  workspace) {
  std::vector<size_t> workspaceIndices;
  auto monitorWorkspace = workspace->monitorWorkspace();
  if (monitorWorkspace)  {
    auto numberOfHistograms = monitorWorkspace->getNumberHistograms();
    for (size_t index = 0; index < numberOfHistograms; ++index) {
        const auto& spectrum = workspace->getSpectrum(index);
        auto spectrumNumber = spectrum.getSpectrumNo();
        auto workspaceIndex = workspace->getIndexFromSpectrumNumber(spectrumNumber);
        workspaceIndices.push_back(workspaceIndex);
    }
  } else {
    auto numberOfHistograms = workspace->getNumberHistograms();
    for (size_t workspaceIndex = 0; workspaceIndex < numberOfHistograms; ++workspaceIndex) {
        auto detector = workspace->getDetector(workspaceIndex);
        if (detector->isMonitor()) {
            workspaceIndices.push_back(workspaceIndex);
        }
    }
  }
  return workspaceIndices;
}

/**
 * We step through the BinEdges object and make sure that each bin width is the same
 * This is done via a pair of moving iterators which are spaced by one BinEdge entry.
 *
 * @param binEdges a handle to the bin edges object.
 * @returns true if the bin edges are linear spaced else false.
 **/
bool areBinEdgesLinearlySpaced(const Mantid::HistogramData::BinEdges& binEdges) {
  auto binEdgesAreLinearlySpaced = true;
  auto lowerBinEdge = binEdges.cbegin();
  auto upperBinEge = binEdges.cbegin();
  ++upperBinEge;
  auto firstBinWidth = *upperBinEge - *lowerBinEdge;
  for (; upperBinEge != binEdges.cend(); ++lowerBinEdge, ++upperBinEge) {
    auto binWidth = *upperBinEge - *lowerBinEdge;
    auto difference = std::abs(binWidth - firstBinWidth);
    // The tolerance of the difference is set such that it must be within a microsecond
    if (difference > 1e-6) {
      binEdgesAreLinearlySpaced = false;
      break;
    }
  }
  return binEdgesAreLinearlySpaced;
}

bool isLinearlySpaced(Mantid::API::MatrixWorkspace_sptr workspace, const std::vector<size_t>& monitorWorkspaceIndices) {
  // Check for each monitor that the spectrum is linearly spaced
  auto isLinearlySpaced = true;
  for (const auto& workspaceIndex : monitorWorkspaceIndices) {
    auto histogram = workspace->histogram(workspaceIndex);
    auto binEdges = histogram.binEdges();
    if (!areBinEdgesLinearlySpaced(binEdges)) {
      isLinearlySpaced = false;
      break;
    }
  }
  return isLinearlySpaced;
}

}

namespace Mantid {
namespace Algorithms {

using Mantid::Kernel::Direction;
using Mantid::API::WorkspaceProperty;

// Register the algorithm into the AlgorithmFactory
DECLARE_ALGORITHM(UnwrapMonitorsInTOF)

//----------------------------------------------------------------------------------------------

/// Algorithms name for identification. @see Algorithm::name
const std::string UnwrapMonitorsInTOF::name() const { return "UnwrapMonitorsInTOF"; }

/// Algorithm's version for identification. @see Algorithm::version
int UnwrapMonitorsInTOF::version() const { return 1; }

/// Algorithm's category for identification. @see Algorithm::category
const std::string UnwrapMonitorsInTOF::category() const {
  return "TODO: FILL IN A CATEGORY";
}

/// Algorithm's summary for use in the GUI and help. @see Algorithm::summary
const std::string UnwrapMonitorsInTOF::summary() const {
  return "TODO: FILL IN A SUMMARY";
}

//----------------------------------------------------------------------------------------------
/** Initialize the algorithm's properties.
 */
void UnwrapMonitorsInTOF::init() {
  declareProperty(
      Kernel::make_unique<WorkspaceProperty<Mantid::API::MatrixWorkspace>>("InputWorkspace", "",
                                                             Direction::Input),
      "An input workspace.");
  declareProperty(
      Kernel::make_unique<WorkspaceProperty<Mantid::API::MatrixWorkspace>>("OutputWorkspace", "",
                                                             Direction::Output),
      "An output workspace.");
  declareProperty<double>("WavelengthMin", specialWavelengthCutoff,  "A lower bound of the wavelength range.");
  declareProperty<double>("WavelengthMax", specialWavelengthCutoff, "An upper bound of the wavelength range.");
}

//----------------------------------------------------------------------------------------------
/** Execute the algorithm.
 */
void UnwrapMonitorsInTOF::exec() {
  // The unwrapping happens in three parts
  // 1. We duplicate the monitor data, by appending the data to itself again. This means
  //    that at there is an interval in which the data is correct
  // 2. Data which is outside of a certain equivalent wavelength range will be set to 0
  Mantid::API::MatrixWorkspace_sptr inputWorkspace = getProperty("InputWorkspace");
  const double lowerWavelengthLimit = getProperty("WavelengthMin");
  const double upperWavelengthLimit = getProperty("WavelengthMax");

  auto outputWorkspace = Mantid::API::MatrixWorkspace_sptr(inputWorkspace->clone());
  const auto workspaceIndices = getWorkspaceIndicesForMonitors(outputWorkspace.get());

  // At the moment we can only support linearly spaced monitor data. If the data is
  // logarithmic we might have to rebin.
  if (!isLinearlySpaced(outputWorkspace, workspaceIndices)) {
    throw std::runtime_error("Monitor data which is not linearly binned is currently not supported.");
  }

  for (const auto& workspaceIndex : workspaceIndices) {
      auto minMaxTof = getMinAndMaxTof(outputWorkspace.get(), workspaceIndex,
                                       lowerWavelengthLimit, upperWavelengthLimit);
      auto points = getPoints(outputWorkspace.get(), workspaceIndex);
      auto counts = getCounts(outputWorkspace.get(), workspaceIndex, minMaxTof, points);
      Mantid::HistogramData::Histogram histogram(points, counts);
      outputWorkspace->setHistogram(workspaceIndex, histogram);
  }
  setProperty("OutputWorkspace", outputWorkspace);
}


/**
 * Check the inputs for invalid values
 * @returns A map with validation warnings.
 */
std::map<std::string, std::string> UnwrapMonitorsInTOF::validateInputs() {
  std::map<std::string, std::string> invalidProperties;
  // The lower wavelength boundary needs to be smaller than the upper wavelength boundary
  double lowerWavelengthLimit = getProperty("WavelengthMin");
  double upperWavelengthLimit = getProperty("WavelengthMax");
  if (lowerWavelengthLimit != specialWavelengthCutoff && lowerWavelengthLimit < 0.0) {
      invalidProperties["WavelengthMin"] = "The lower wavelength limit must be set to a positive value.";
  }

  if (upperWavelengthLimit != specialWavelengthCutoff && upperWavelengthLimit < 0.0) {
      invalidProperties["WavelengthMax"] = "The upper wavelength limit must be set to a positive value.";
  }

  if (lowerWavelengthLimit != specialWavelengthCutoff  && upperWavelengthLimit != specialWavelengthCutoff 
    && lowerWavelengthLimit >= upperWavelengthLimit) {
      invalidProperties["WavelengthMin"] = "The lower wavelength limit must be smaller than the upper wavelnegth limit.";
      invalidProperties["WavelengthMax"] = "The lower wavelength limit must be smaller than the upper wavelnegth limit.";
  }

  return invalidProperties;
}


} // namespace Algorithms
} // namespace Mantid
