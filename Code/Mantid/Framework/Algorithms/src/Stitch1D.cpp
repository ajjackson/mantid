/*WIKI*
TODO: Enter a full wiki-markup description of your algorithm here. You can then use the Build/wiki_maker.py script to generate your full wiki page.
*WIKI*/

#include "MantidAlgorithms/Stitch1D.h"
#include "MantidAPI/WorkspaceProperty.h"
#include "MantidAPI/MatrixWorkspace.h"
#include "MantidAPI/WorkspaceValidators.h"
#include "MantidKernel/ArrayProperty.h"
#include "MantidKernel/PropertyWithValue.h"
#include "MantidKernel/RebinParamsValidator.h"

#include <boost/make_shared.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <algorithm>

using namespace Mantid::Kernel;
using namespace Mantid::API;
using Mantid::MantidVec;

namespace
{

  typedef boost::tuple<double, double> MinMaxTuple;
  MinMaxTuple calculateXIntersection(MatrixWorkspace_sptr lhsWS, MatrixWorkspace_sptr rhsWS)
  {
    MantidVec lhs_x = lhsWS->readX(0);
    MantidVec rhs_x = rhsWS->readX(0);
    return MinMaxTuple(rhs_x.front(), lhs_x.back());
  }

  bool isNonzero (double i)
  {
    return (0 != i);
  }
}

namespace Mantid
{
  namespace Algorithms
  {

    // Register the algorithm into the AlgorithmFactory
    DECLARE_ALGORITHM(Stitch1D)

    //----------------------------------------------------------------------------------------------
    /** Constructor
    */
    Stitch1D::Stitch1D()
    {
    }

    //----------------------------------------------------------------------------------------------
    /** Destructor
    */
    Stitch1D::~Stitch1D()
    {
    }

    //----------------------------------------------------------------------------------------------
    /// Algorithm's name for identification. @see Algorithm::name
    const std::string Stitch1D::name() const
    {
      return "Stitch1D";
    }
    ;

    /// Algorithm's version for identification. @see Algorithm::version
    int Stitch1D::version() const
    {
      return 4;
    }
    ;

    /// Algorithm's category for identification. @see Algorithm::category
    const std::string Stitch1D::category() const
    {
      return "Reflectometry";
    }

    //----------------------------------------------------------------------------------------------
    /// Sets documentation strings for this algorithm
    void Stitch1D::initDocs()
    {
      this->setWikiSummary("Stitches single histogram matrix workspaces together");
      this->setOptionalMessage(this->getWikiSummary());
    }

    //----------------------------------------------------------------------------------------------
    /** Initialize the algorithm's properties.
    */
    void Stitch1D::init()
    {
      Kernel::IValidator_sptr histogramValidator = boost::make_shared<HistogramValidator>();

      declareProperty(
        new WorkspaceProperty<MatrixWorkspace>("LHSWorkspace", "", Direction::Input,
        histogramValidator->clone()), "LHS input workspace.");
      declareProperty(
        new WorkspaceProperty<MatrixWorkspace>("RHSWorkspace", "", Direction::Input,
        histogramValidator->clone()), "RHS input workspace.");
      declareProperty(new WorkspaceProperty<MatrixWorkspace>("OutputWorkspace", "", Direction::Output),
        "Output stitched workspace.");
      declareProperty(
        new PropertyWithValue<double>("StartOverlap", Mantid::EMPTY_DBL(), Direction::Input),
        "Start overlap x-value in units of x-axis. Optional.");
      declareProperty(new PropertyWithValue<double>("EndOverlap", Mantid::EMPTY_DBL(), Direction::Input),
        "End overlap x-value in units of x-axis. Optional.");
      declareProperty(
        new ArrayProperty<double>("Params", boost::make_shared<RebinParamsValidator>(true)),
        "Rebinning Parameters. See Rebin for format. If only a single value is provided, start and end are taken from input workspaces.");
      declareProperty(new PropertyWithValue<bool>("ScaleRHSWorkspace", true, Direction::Input),
        "Scaling either with respect to workspace 1 or workspace 2");
      declareProperty(new PropertyWithValue<bool>("UseManualScaleFactor", false, Direction::Input),
        "True to use a provided value for the scale factor.");
      declareProperty(
        new PropertyWithValue<double>("OutScaleFactor", Mantid::EMPTY_DBL(), Direction::Output),
        "The actual used value for the scaling factor.");

    }

    double Stitch1D::getStartOverlap(const double& intesectionMin, const double& intesectionMax) const
    {
      Property* startOverlapProp = this->getProperty("StartOverlap");
      double startOverlapVal = this->getProperty("StartOverlap");
      const bool startOverlapBeyondRange = (startOverlapVal < intesectionMin)
        || (startOverlapVal > intesectionMax);
      if (startOverlapProp->isDefault() || startOverlapBeyondRange)
      {
        if (!startOverlapProp->isDefault() && startOverlapBeyondRange)
        {
          char message[200];
          std::sprintf(message,
            "StartOverlap is outside range at %0.4f, Min is %0.4f, Max is %0.4f . Forced to be: %0.4f",
            startOverlapVal, intesectionMin, intesectionMax, intesectionMin);
          g_log.warning(std::string(message));
        }
        startOverlapVal = intesectionMin;
        std::stringstream buffer;
        buffer << "StartOverlap calculated to be: " << startOverlapVal;
        g_log.information(buffer.str());
      }
      return startOverlapVal;
    }

    double Stitch1D::getEndOverlap(const double& intesectionMin, const double& intesectionMax) const
    {
      Property* endOverlapProp = this->getProperty("EndOverlap");
      double endOverlapVal = this->getProperty("EndOverlap");
      const bool endOverlapBeyondRange = (endOverlapVal < intesectionMin)
        || (endOverlapVal > intesectionMax);
      if (endOverlapProp->isDefault() || endOverlapBeyondRange)
      {
        if (!endOverlapProp->isDefault() && endOverlapBeyondRange)
        {
          char message[200];
          std::sprintf(message,
            "EndOverlap is outside range at %0.4f, Min is %0.4f, Max is %0.4f . Forced to be: %0.4f",
            endOverlapVal, intesectionMin, intesectionMax, intesectionMin);
          g_log.warning(std::string(message));
        }
        endOverlapVal = intesectionMax;
        std::stringstream buffer;
        buffer << "EndOverlap calculated to be: " << endOverlapVal;
        g_log.information(buffer.str());
      }
      return endOverlapVal;
    }

    MantidVec Stitch1D::getRebinParams(MatrixWorkspace_sptr& lhsWS, MatrixWorkspace_sptr& rhsWS) const
    {
      MantidVec inputParams = this->getProperty("Params");
      Property* prop = this->getProperty("Params");
      const bool areParamsDefault = prop->isDefault();

      const MantidVec& lhsX = lhsWS->readX(0);
      auto it = std::min_element(lhsX.begin(), lhsX.end());
      const double minLHSX = *it;

      const MantidVec& rhsX = rhsWS->readX(0);
      it = std::max_element(rhsX.begin(), rhsX.end());
      const double maxRHSX = *it;

      MantidVec result;
      if (areParamsDefault)
      {
        MantidVec calculatedParams;

        const double calculatedStep = (maxRHSX - minLHSX) / 100;
        calculatedParams.push_back(minLHSX);
        calculatedParams.push_back(calculatedStep);
        calculatedParams.push_back(maxRHSX);
        result = calculatedParams;
      }
      else
      {
        if (inputParams.size() == 1)
        {
          MantidVec calculatedParams;
          calculatedParams.push_back(minLHSX);
          calculatedParams.push_back(inputParams.front()); // Use the step supplied.
          calculatedParams.push_back(maxRHSX);
          result = calculatedParams;
        }
        else
        {
          result = inputParams; // user has provided params. Use those.
        }
      }
      return result;
    }

    MatrixWorkspace_sptr Stitch1D::rebin(MatrixWorkspace_sptr& input, const MantidVec& params)
    {
      auto rebin = this->createChildAlgorithm("Rebin");
      rebin->setProperty("InputWorkspace", input);
      rebin->setProperty("Params", params);
      rebin->execute();
      MatrixWorkspace_sptr outWS = rebin->getProperty("OutputWorkspace");
      return outWS;
    }

    MatrixWorkspace_sptr Stitch1D::integration(MatrixWorkspace_sptr& input, const double& start, const double& stop)
    {
      auto integration = this->createChildAlgorithm("Integration");
      integration->setProperty("InputWorkspace", input);
      integration->setProperty("RangeLower", start);
      integration->setProperty("RangeUpper", stop);
      integration->execute();
      MatrixWorkspace_sptr outWS = integration->getProperty("OutputWorkspace");
      return outWS;
    }

    MatrixWorkspace_sptr Stitch1D::multiplyRange(MatrixWorkspace_sptr& input, const int& startBin, const int& endBin, const double& factor)
    {
      auto multiplyRange = this->createChildAlgorithm("MultiplyRange");
      multiplyRange->setProperty("InputWorkspace", input);
      multiplyRange->setProperty("StartBin", startBin);
      multiplyRange->setProperty("EndBin", endBin);
      multiplyRange->setProperty("Factor", factor);
      multiplyRange->execute();
      MatrixWorkspace_sptr outWS = multiplyRange->getProperty("OutputWorkspace");
      return outWS;
    }

    MatrixWorkspace_sptr Stitch1D::multiplyRange(MatrixWorkspace_sptr& input, const int& startBin, const double& factor)
    {
      auto multiplyRange = this->createChildAlgorithm("MultiplyRange");
      multiplyRange->setProperty("InputWorkspace", input);
      multiplyRange->setProperty("StartBin", startBin);
      multiplyRange->setProperty("Factor", factor);
      multiplyRange->execute();
      MatrixWorkspace_sptr outWS = multiplyRange->getProperty("OutputWorkspace");
      return outWS;
    }

    MatrixWorkspace_sptr Stitch1D::weightedMean(MatrixWorkspace_sptr& inOne, MatrixWorkspace_sptr& inTwo)
    {
      auto weightedMean = this->createChildAlgorithm("WeightedMean");
      weightedMean->setProperty("InputWorkspace1", inOne);
      weightedMean->setProperty("InputWorkspace2", inTwo);
      weightedMean->execute();
      MatrixWorkspace_sptr outWS = weightedMean->getProperty("OutputWorkspace");
      return outWS;
    }

    MatrixWorkspace_sptr Stitch1D::singleValueWS(double val)
    {
      auto singleValueWS = this->createChildAlgorithm("CreateSingleValuedWorkspace");
      singleValueWS->setProperty("DataValue", val);
      singleValueWS->execute();
      MatrixWorkspace_sptr outWS = singleValueWS->getProperty("OutputWorkspace");
      return outWS;
    }

    boost::tuple<int, int> Stitch1D::findStartEndIndexes(double startOverlap, double endOverlap, MatrixWorkspace_sptr& workspace)
    {
      int a1 = static_cast<int>(workspace->binIndexOf(startOverlap));
      int a2 = static_cast<int>(workspace->binIndexOf(endOverlap));
      if (a1 == a2)
      {
        throw std::runtime_error("The Params you have provided for binning yield a workspace in which start and end overlap appear in the same bin. Make binning finer via input Params.");
      }
      return boost::tuple<int,int>(a1,a2);

    }

    bool Stitch1D::hasNonzeroErrors(MatrixWorkspace_sptr& ws) const
    {
      size_t ws_size = ws->getNumberHistograms();
      for (size_t i = 0; i < ws_size; ++i)
      {
        auto e = ws->readE(i);
        std::vector<double, std::allocator<double> >::iterator error = std::find_if(e.begin(), e.end(), isNonzero);
        if (error != e.end())
        {
          return true;
        }
      }
      return false;
    }

    //----------------------------------------------------------------------------------------------
    /** Execute the algorithm.
    */
    void Stitch1D::exec()
    {
      MatrixWorkspace_sptr rhsWS = this->getProperty("RHSWorkspace");
      MatrixWorkspace_sptr lhsWS = this->getProperty("LHSWorkspace");
      const MinMaxTuple intesectionXRegion = calculateXIntersection(lhsWS, rhsWS);

      const double intersectionMin = intesectionXRegion.get<0>();
      const double intersectionMax = intesectionXRegion.get<1>();

      const double startOverlap = getStartOverlap(intersectionMin, intersectionMax);
      const double endOverlap = getEndOverlap(intersectionMin, intersectionMax);

      if (startOverlap > endOverlap)
      {
        std::string message =
          boost::str(
          boost::format(
          "Stitch1D cannot have a StartOverlap > EndOverlap. StartOverlap: %0.9f, EndOverlap: %0.9f")
          % startOverlap % endOverlap);
        throw std::runtime_error(message);
      }

      MantidVec params = getRebinParams(lhsWS, rhsWS);

      const double& xMin = params.front();
      const double& xMax = params.back();

      if (startOverlap < xMin)
      {
        std::string message =
          boost::str(
          boost::format(
          "Stitch1D StartOverlap is outside the available X range after rebinning. StartOverlap: %10.9f, X min: %10.9f")
          % startOverlap % xMin);

        throw std::runtime_error(message);
      }
      if (endOverlap > xMax)
      {
        std::string message =
          boost::str(
          boost::format(
          "Stitch1D EndOverlap is outside the available X range after rebinning. EndOverlap: %10.9f, X max: %10.9f")
          % endOverlap % xMax);

        throw std::runtime_error(message);
      }

      auto rebinnedLHS = rebin(lhsWS, params);
      auto rebinnedRHS = rebin(rhsWS, params);

      boost::tuple<int,int> startEnd = findStartEndIndexes(startOverlap, endOverlap, rebinnedLHS);

      //manualscalefactor if

      auto rhsOverlapIntegrated = integration(rebinnedRHS, startOverlap, endOverlap);
      auto lhsOverlapIntegrated = integration(rebinnedLHS, startOverlap, endOverlap);

      auto y1 = lhsOverlapIntegrated->readY(0);
      auto y2 = lhsOverlapIntegrated->readY(0);
      double scaleFactor = 0;
      if(rhsOverlapIntegrated)
      {
        MatrixWorkspace_sptr ratio = lhsOverlapIntegrated/rhsOverlapIntegrated;
        rebinnedRHS =  rebinnedRHS * ratio;
        scaleFactor = y1[0]/y2[0];
      }
      else
      {
        MatrixWorkspace_sptr ratio = rhsOverlapIntegrated/lhsOverlapIntegrated;
        rebinnedLHS =  rebinnedLHS * ratio;
        scaleFactor = y2[0]/y1[0];
      }

      //manualscalefactor end if

      int a1 = boost::tuples::get<0>(startEnd);
      int a2 = boost::tuples::get<1>(startEnd);

      // Mask out everything BUT the overlap region as a new workspace.
      MatrixWorkspace_sptr overlap1 = multiplyRange(rebinnedLHS,0,a1,0);
      overlap1 = multiplyRange(overlap1,a2,0);

      // Mask out everything BUT the overlap region as a new workspace.
      MatrixWorkspace_sptr overlap2 = multiplyRange(rebinnedRHS,0,a1,0);
      overlap2 = multiplyRange(overlap2,a2,0);

      // Mask out everything AFTER the start of the overlap region
      rebinnedLHS = multiplyRange(rebinnedLHS,a1 + 1,0);

      // Mask out everything BEFORE the end of the overlap region
      rebinnedRHS = multiplyRange(rebinnedRHS,0,a2-1,0);

      // Calculate a weighted mean for the overlap region

      MatrixWorkspace_sptr overlapave;
      if (hasNonzeroErrors(overlap1) && hasNonzeroErrors(overlap2))
      {
        overlapave = weightedMean(overlap1, overlap2);
      }
      else
      {
        g_log.information("Using un-weighted mean for Stitch1D overlap mean");
        MatrixWorkspace_sptr sum = overlap1 + overlap2;
        MatrixWorkspace_sptr denominator = singleValueWS(2.0);
        overlapave = sum / denominator;
      }

      MatrixWorkspace_sptr result = rebinnedLHS + overlapave + rebinnedRHS;

      setProperty("OutputWorkspace", result);
      setProperty("OutScaleFactor", scaleFactor);

    }

  } // namespace Algorithms
} // namespace Mantid
