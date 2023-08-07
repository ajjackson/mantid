// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cxxtest/TestSuite.h>

#include "MantidAPI/AnalysisDataService.h"
#include "MantidAPI/RefAxis.h"
#include "MantidAPI/ScopedWorkspace.h"
#include "MantidAPI/SpectraAxis.h"
#include "MantidAPI/WorkspaceProperty.h"
#include "MantidAlgorithms/CreateWorkspace.h"
#include "MantidAlgorithms/MaskBins.h"
#include "MantidAlgorithms/Rebin.h"
#include "MantidDataObjects/EventWorkspace.h"
#include "MantidDataObjects/Workspace2D.h"
#include "MantidFrameworkTestHelpers/WorkspaceCreationHelper.h"
#include "MantidHistogramData/LinearGenerator.h"

#include <numeric>

using namespace Mantid;
using namespace Mantid::Kernel;
using namespace Mantid::DataObjects;
using namespace Mantid::API;
using namespace Mantid::Algorithms;
using Mantid::HistogramData::BinEdges;
using Mantid::HistogramData::Counts;
using Mantid::HistogramData::CountStandardDeviations;

class RebinTest : public CxxTest::TestSuite {
public:
  double BIN_DELTA;
  int NUMPIXELS, NUMBINS;

  // This pair of boilerplate methods prevent the suite being created statically
  // This means the constructor isn't called when running other tests
  static RebinTest *createSuite() { return new RebinTest(); }
  static void destroySuite(RebinTest *suite) { delete suite; }

  RebinTest() {
    BIN_DELTA = 2.0;
    NUMPIXELS = 20;
    NUMBINS = 50;
  }

  void testworkspace1D_dist() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    test_in1D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_in1D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    // Check it fails if "Params" property not set
    TS_ASSERT_THROWS(rebin.execute(), const std::runtime_error &);
    TS_ASSERT(!rebin.isExecuted());
    // Now set the property
    rebin.setPropertyValue("Params", "1.5,2.0,20,-0.1,30,1.0,35");
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());
    MatrixWorkspace_sptr rebindata = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out");
    auto &outX = rebindata->x(0);
    auto &outY = rebindata->y(0);
    auto &outE = rebindata->e(0);

    TS_ASSERT_DELTA(outX[7], 15.5, 0.000001);
    TS_ASSERT_DELTA(outY[7], 3.0, 0.000001);
    TS_ASSERT_DELTA(outE[7], sqrt(4.5) / 2.0, 0.000001);

    TS_ASSERT_DELTA(outX[12], 24.2, 0.000001);
    TS_ASSERT_DELTA(outY[12], 3.0, 0.000001);
    TS_ASSERT_DELTA(outE[12], sqrt(5.445) / 2.42, 0.000001);

    TS_ASSERT_DELTA(outX[17], 32.0, 0.000001);
    TS_ASSERT_DELTA(outY[17], 3.0, 0.000001);
    TS_ASSERT_DELTA(outE[17], sqrt(2.25), 0.000001);
    bool dist = rebindata->isDistribution();
    TS_ASSERT(dist);
    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out");
  }

  void testworkspace1D_nondist() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_in1D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    rebin.setPropertyValue("Params", "1.5,2.0,20,-0.1,30,1.0,35");
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());
    MatrixWorkspace_sptr rebindata = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out");

    auto &outX = rebindata->x(0);
    auto &outY = rebindata->y(0);
    auto &outE = rebindata->e(0);

    TS_ASSERT_DELTA(outX[7], 15.5, 0.000001);
    TS_ASSERT_DELTA(outY[7], 8.0, 0.000001);
    TS_ASSERT_DELTA(outE[7], sqrt(8.0), 0.000001);
    TS_ASSERT_DELTA(outX[12], 24.2, 0.000001);
    TS_ASSERT_DELTA(outY[12], 9.68, 0.000001);
    TS_ASSERT_DELTA(outE[12], sqrt(9.68), 0.000001);
    TS_ASSERT_DELTA(outX[17], 32, 0.000001);
    TS_ASSERT_DELTA(outY[17], 4.0, 0.000001);
    TS_ASSERT_DELTA(outE[17], sqrt(4.0), 0.000001);
    bool dist = rebindata->isDistribution();
    TS_ASSERT(!dist);
    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out");
  }

  void test_make_three_params() {
    // test that if only a binwidth is given, will create an xmin, xmax from xstep
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_in1D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    rebin.setPropertyValue("Params", "2.0");
    rebin.setPropertyValue("BinningMode", "Linear");
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());

    MatrixWorkspace_sptr inputdata = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_in1D");
    MatrixWorkspace_sptr rebindata = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out");

    auto &outX = rebindata->x(0);
    double xmin = outX[0];
    double xmax = outX[outX.size() - 1];
    std::vector<double> rbParams = rebin.getProperty("Params");
    TS_ASSERT_EQUALS(rbParams.size(), 3);
    TS_ASSERT_EQUALS(xmin, rbParams[0]);
    TS_ASSERT_EQUALS(2.0, rbParams[1]);
    TS_ASSERT_EQUALS(xmax, rbParams[2]);

    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out");
  }

  void testworkspace1D_logarithmic_binning() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    test_in1D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_in1D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    // Check it fails if "Params" property not set
    TS_ASSERT_THROWS(rebin.execute(), const std::runtime_error &);
    TS_ASSERT(!rebin.isExecuted());
    // Now set the property
    rebin.setPropertyValue("Params", "1.0,-1.0,1000.0");
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());
    MatrixWorkspace_sptr rebindata = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out");
    auto &outX = rebindata->x(0);

    TS_ASSERT_EQUALS(outX.size(), 11);
    TS_ASSERT_DELTA(outX[0], 1.0, 1e-5);
    TS_ASSERT_DELTA(outX[1], 2.0, 1e-5);
    TS_ASSERT_DELTA(outX[2], 4.0, 1e-5); // and so on...
    TS_ASSERT_DELTA(outX[10], 1000.0, 1e-5);

    bool dist = rebindata->isDistribution();
    TS_ASSERT(dist);

    TS_ASSERT(checkBinWidthMonotonic(rebindata, Monotonic::INCREASE));

    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out");
  }

  // Check that logarithmic binning is performed if the BinningMode is set, even if positive binwidth
  // Run once in the default manner, then run again with BinningMode set, and compare.
  void testworkspace1D_logarithmic_binning_modeset() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    test_in1D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin_default;
    rebin_default.initialize();
    rebin_default.setPropertyValue("InputWorkspace", "test_in1D");
    rebin_default.setPropertyValue("OutputWorkspace", "test_out_default");
    rebin_default.setPropertyValue("Params", "1.0,-1.0,1000.0");
    TS_ASSERT(rebin_default.execute());
    TS_ASSERT(rebin_default.isExecuted());
    MatrixWorkspace_sptr rebindata_default =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_default");
    auto &outX_default = rebindata_default->x(0);

    // now run again but with positive step, using logarithmic mode
    Rebin rebin_again;
    rebin_again.initialize();
    rebin_again.setPropertyValue("InputWorkspace", "test_in1D");
    rebin_again.setPropertyValue("OutputWorkspace", "test_out_again");
    rebin_again.setPropertyValue("Params", "1.0,1.0,1000.0");
    rebin_again.setPropertyValue("BinningMode", "Logarithmic");
    TS_ASSERT(rebin_again.execute());
    TS_ASSERT(rebin_again.isExecuted());
    MatrixWorkspace_sptr rebindata_again =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_again");
    auto &outX_again = rebindata_again->x(0);

    TS_ASSERT_EQUALS(outX_default.size(), outX_again.size());
    for (size_t i = 0; i < outX_default.size(); i++) {
      TS_ASSERT_EQUALS(outX_default[i], outX_again[i]);
    }
    TS_ASSERT(checkBinWidthMonotonic(rebindata_again, Monotonic::INCREASE));

    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out_default");
    AnalysisDataService::Instance().remove("test_out_again");
  }

  // Check that reverse logarithmic binning is performed if the BinningMode is set,
  // even if positive binwidth and even if UseReverseLogarithmic set to false.
  // Run once in the default manner, then run again with BinningMode set, and compare.
  void testworkspace1D_reverse_logarithmic_binning_modeset() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    test_in1D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin_default;
    rebin_default.initialize();
    rebin_default.setPropertyValue("InputWorkspace", "test_in1D");
    rebin_default.setPropertyValue("OutputWorkspace", "test_out_default");
    rebin_default.setProperty("Params", "1.0,-1.0,1000.0");
    rebin_default.setProperty("UseReverseLogarithmic", true);
    TS_ASSERT(rebin_default.execute());
    TS_ASSERT(rebin_default.isExecuted());
    MatrixWorkspace_sptr rebindata_default =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_default");
    auto &outX_default = rebindata_default->x(0);

    // now run again but with positive step, using logarithmic mode
    Rebin rebin_again;
    rebin_again.initialize();
    rebin_again.setPropertyValue("InputWorkspace", "test_in1D");
    rebin_again.setPropertyValue("OutputWorkspace", "test_out_again");
    rebin_again.setProperty("Params", "1.0,1.0,1000.0");
    rebin_default.setProperty("UseReverseLogarithmic", false);
    rebin_again.setProperty("BinningMode", "ReverseLogarithmic");
    TS_ASSERT(rebin_again.execute());
    TS_ASSERT(rebin_again.isExecuted());
    MatrixWorkspace_sptr rebindata_again =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_again");
    auto &outX_again = rebindata_again->x(0);

    TS_ASSERT_EQUALS(outX_default.size(), outX_again.size());
    for (size_t i = 0; i < outX_default.size(); i++) {
      TS_ASSERT_EQUALS(outX_default[i], outX_again[i]);
    }
    TS_ASSERT(checkBinWidthMonotonic(rebindata_again, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out_default");
    AnalysisDataService::Instance().remove("test_out_again");
  }

  void test_bad_binning_modeset() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    test_in1D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setRethrows(true);
    rebin.setPropertyValue("InputWorkspace", "test_in1D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    rebin.setPropertyValue("Params", "1.0,1.0,1000.0");
    rebin.setProperty("BinningMode", "FunkyCrosswaysBinning"); // this is not a known binning mode
    TS_ASSERT_THROWS_ANYTHING(rebin.execute(););
    TS_ASSERT(!rebin.isExecuted());

    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out");
  }

  void test_failure_bad_log_binning_endpoints() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    test_in1D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in1D", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setRethrows(true);
    rebin.setPropertyValue("InputWorkspace", "test_in1D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    rebin.setPropertyValue("Params", "-1.0,1.0,1000.0");
    rebin.setProperty("BinningMode", "Logarithmic");
    TS_ASSERT_THROWS(rebin.execute(), const std::runtime_error &);
    TS_ASSERT(!rebin.isExecuted());

    AnalysisDataService::Instance().remove("test_in1D");
    AnalysisDataService::Instance().remove("test_out");
  }

  void testworkspace2D_dist() {
    Workspace2D_sptr test_in2D = Create2DWorkspace(50, 20);
    test_in2D->setDistribution(true);
    AnalysisDataService::Instance().add("test_in2D", test_in2D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_in2D");
    rebin.setPropertyValue("OutputWorkspace", "test_out");
    rebin.setPropertyValue("Params", "1.5,2.0,20,-0.1,30,1.0,35");
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());
    MatrixWorkspace_sptr rebindata = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out");

    auto &outX = rebindata->x(5);
    auto &outY = rebindata->y(5);
    auto &outE = rebindata->e(5);
    TS_ASSERT_DELTA(outX[7], 15.5, 0.000001);
    TS_ASSERT_DELTA(outY[7], 3.0, 0.000001);
    TS_ASSERT_DELTA(outE[7], sqrt(4.5) / 2.0, 0.000001);

    TS_ASSERT_DELTA(outX[12], 24.2, 0.000001);
    TS_ASSERT_DELTA(outY[12], 3.0, 0.000001);
    TS_ASSERT_DELTA(outE[12], sqrt(5.445) / 2.42, 0.000001);

    TS_ASSERT_DELTA(outX[17], 32.0, 0.000001);
    TS_ASSERT_DELTA(outY[17], 3.0, 0.000001);
    TS_ASSERT_DELTA(outE[17], sqrt(2.25), 0.000001);
    bool dist = rebindata->isDistribution();
    TS_ASSERT(dist);

    // Test the axes are of the correct type
    TS_ASSERT_EQUALS(rebindata->axes(), 2);
    TS_ASSERT(dynamic_cast<RefAxis *>(rebindata->getAxis(0)));
    TS_ASSERT(dynamic_cast<SpectraAxis *>(rebindata->getAxis(1)));

    AnalysisDataService::Instance().remove("test_in2D");
    AnalysisDataService::Instance().remove("test_out");
  }

  void do_test_EventWorkspace(EventType eventType, bool inPlace, bool PreserveEvents, bool expectOutputEvent) {
    // Two events per bin
    EventWorkspace_sptr test_in = WorkspaceCreationHelper::createEventWorkspace2(50, 100);
    test_in->switchEventType(eventType);

    std::string inName("test_inEvent");
    std::string outName("test_inEvent_output");
    if (inPlace)
      outName = inName;

    AnalysisDataService::Instance().addOrReplace(inName, test_in);
    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", inName);
    rebin.setPropertyValue("OutputWorkspace", outName);
    rebin.setPropertyValue("Params", "0.0,4.0,100");
    rebin.setProperty("PreserveEvents", PreserveEvents);
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());

    MatrixWorkspace_sptr outWS;
    EventWorkspace_sptr eventOutWS;
    TS_ASSERT_THROWS_NOTHING(
        outWS = std::dynamic_pointer_cast<MatrixWorkspace>(AnalysisDataService::Instance().retrieve(outName)));
    TS_ASSERT(outWS);
    if (!outWS)
      return;

    // Is the output gonna be events?
    if (expectOutputEvent) {
      eventOutWS = std::dynamic_pointer_cast<EventWorkspace>(outWS);
      TS_ASSERT(eventOutWS);
      if (!eventOutWS)
        return;
      TS_ASSERT_EQUALS(eventOutWS->getNumberEvents(), 50 * 100 * 2);
      // Check that it is the same workspace
      if (inPlace)
        TS_ASSERT(eventOutWS == test_in);
    }

    auto &X = outWS->x(0);
    auto &Y = outWS->y(0);
    auto &E = outWS->e(0);

    TS_ASSERT_EQUALS(X.size(), 26);
    TS_ASSERT_DELTA(X[0], 0.0, 1e-5);
    TS_ASSERT_DELTA(X[1], 4.0, 1e-5);
    TS_ASSERT_DELTA(X[2], 8.0, 1e-5);

    TS_ASSERT_EQUALS(Y.size(), 25);
    TS_ASSERT_DELTA(Y[0], 8.0, 1e-5);
    TS_ASSERT_DELTA(Y[1], 8.0, 1e-5);
    TS_ASSERT_DELTA(Y[2], 8.0, 1e-5);

    TS_ASSERT_EQUALS(E.size(), 25);
    TS_ASSERT_DELTA(E[0], sqrt(8.0), 1e-5);
    TS_ASSERT_DELTA(E[1], sqrt(8.0), 1e-5);

    // Test the axes are of the correct type
    TS_ASSERT_EQUALS(outWS->axes(), 2);
    TS_ASSERT(dynamic_cast<RefAxis *>(outWS->getAxis(0)));
    TS_ASSERT(dynamic_cast<SpectraAxis *>(outWS->getAxis(1)));

    AnalysisDataService::Instance().remove(inName);
    AnalysisDataService::Instance().remove(outName);
  }

  void testEventWorkspace_InPlace_PreserveEvents() { do_test_EventWorkspace(TOF, true, true, true); }

  void testEventWorkspace_InPlace_PreserveEvents_weighted() { do_test_EventWorkspace(WEIGHTED, true, true, true); }

  void testEventWorkspace_InPlace_PreserveEvents_weightedNoTime() {
    do_test_EventWorkspace(WEIGHTED_NOTIME, true, true, true);
  }

  void testEventWorkspace_InPlace_NoPreserveEvents() { do_test_EventWorkspace(TOF, true, false, false); }

  void testEventWorkspace_InPlace_NoPreserveEvents_weighted() { do_test_EventWorkspace(WEIGHTED, true, false, false); }

  void testEventWorkspace_InPlace_NoPreserveEvents_weightedNoTime() {
    do_test_EventWorkspace(WEIGHTED_NOTIME, true, false, false);
  }

  void testEventWorkspace_NotInPlace_NoPreserveEvents() { do_test_EventWorkspace(TOF, false, false, false); }

  void testEventWorkspace_NotInPlace_NoPreserveEvents_weighted() {
    do_test_EventWorkspace(WEIGHTED, false, false, false);
  }

  void testEventWorkspace_NotInPlace_NoPreserveEvents_weightedNoTime() {
    do_test_EventWorkspace(WEIGHTED_NOTIME, false, false, false);
  }

  void testEventWorkspace_NotInPlace_PreserveEvents() { do_test_EventWorkspace(TOF, false, true, true); }

  void testEventWorkspace_NotInPlace_PreserveEvents_weighted() { do_test_EventWorkspace(WEIGHTED, false, true, true); }

  void testEventWorkspace_NotInPlace_PreserveEvents_weightedNoTime() {
    do_test_EventWorkspace(WEIGHTED_NOTIME, false, true, true);
  }

  void testRebinPointData() {
    Workspace2D_sptr input = Create1DWorkspace(51);
    AnalysisDataService::Instance().add("test_RebinPointDataInput", input);

    Mantid::API::Algorithm_sptr ctpd = Mantid::API::AlgorithmFactory::Instance().create("ConvertToPointData", 1);
    ctpd->initialize();
    ctpd->setPropertyValue("InputWorkspace", "test_RebinPointDataInput");
    ctpd->setPropertyValue("OutputWorkspace", "test_RebinPointDataInput");
    ctpd->execute();

    Mantid::API::Algorithm_sptr reb = Mantid::API::AlgorithmFactory::Instance().create("Rebin", 1);
    reb->initialize();
    TS_ASSERT_THROWS_NOTHING(reb->setPropertyValue("InputWorkspace", "test_RebinPointDataInput"));
    reb->setPropertyValue("OutputWorkspace", "test_RebinPointDataOutput");
    reb->setPropertyValue("Params", "7,0.75,23");
    TS_ASSERT_THROWS_NOTHING(reb->execute());

    TS_ASSERT(reb->isExecuted());

    MatrixWorkspace_sptr outWS = std::dynamic_pointer_cast<MatrixWorkspace>(
        AnalysisDataService::Instance().retrieve("test_RebinPointDataOutput"));

    TS_ASSERT(!outWS->isHistogramData());
    TS_ASSERT_EQUALS(outWS->getNumberHistograms(), 1);

    TS_ASSERT_EQUALS(outWS->x(0)[0], 7.3750);
    TS_ASSERT_EQUALS(outWS->x(0)[10], 14.8750);
    TS_ASSERT_EQUALS(outWS->x(0)[20], 22.3750);

    AnalysisDataService::Instance().remove("test_RebinPointDataInput");
    AnalysisDataService::Instance().remove("test_RebinPointDataOutput");
  }

  void testMaskedBinsDist() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(50);
    AnalysisDataService::Instance().add("test_Rebin_mask_dist", test_in1D);
    test_in1D->setDistribution(true);
    maskFirstBins("test_Rebin_mask_dist", "test_Rebin_masked_ws", 10.0);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_masked_ws");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_masked_ws");
    rebin.setPropertyValue("Params", "1.5,3.0,12,-0.1,30");
    TS_ASSERT(rebin.execute());
    TS_ASSERT(rebin.isExecuted());
    MatrixWorkspace_sptr rebindata =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_masked_ws");
    auto &outX = rebindata->x(0);
    auto &outY = rebindata->y(0);

    MatrixWorkspace_sptr input = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_mask_dist");
    auto &inX = input->x(0);
    auto &inY = input->y(0);

    const MatrixWorkspace::MaskList &mask = rebindata->maskedBins(0);

    // turn the mask list into an array like the Y values
    MantidVec weights(outY.size(), 0);
    for (auto it : mask) {
      weights[it.first] = it.second;
    }

    // the degree of masking must be the same as the reduction in the y-value,
    // for distributions, this is the easy case
    for (size_t i = 0; i < outY.size(); ++i) {
      size_t inBin = std::lower_bound(inX.begin(), inX.end(), outX[i]) - inX.begin();
      if (inBin < inX.size() - 2) {
        TS_ASSERT_DELTA(outY[i] / inY[inBin], 1 - weights[i], 0.000001);
      }
    }
    // the above formula tests the criterian that must be true for masking,
    // however we need some more specific tests incase outY.empty() or something
    TS_ASSERT_DELTA(outY[1], 0.0, 0.000001)
    TS_ASSERT_DELTA(outY[2], 0.25, 0.000001)
    TS_ASSERT_DELTA(outY[3], 3.0, 0.000001)
    TS_ASSERT_DELTA(weights[2], 1 - (0.25 / 3.0), 0.000001)

    TS_ASSERT(rebindata->isDistribution());
    AnalysisDataService::Instance().remove("test_Rebin_mask_dist");
    AnalysisDataService::Instance().remove("test_Rebin_masked_ws");
  }

  void testMaskedBinsIntegratedCounts() {
    Workspace2D_sptr test_in1D = Create1DWorkspace(51);
    test_in1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_mask_raw", test_in1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_mask_raw");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_unmasked");
    rebin.setPropertyValue("Params", "1.5,3.0,12,-0.1,30");
    rebin.execute();

    MatrixWorkspace_sptr input = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_unmasked");
    auto &inX = input->x(0);
    auto &inY = input->y(0);

    maskFirstBins("test_Rebin_mask_raw", "test_Rebin_masked_ws", 10.0);

    rebin.setPropertyValue("InputWorkspace", "test_Rebin_masked_ws");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_masked_ws");
    rebin.setPropertyValue("Params", "1.5,3.0,12,-0.1,30");
    rebin.execute();
    MatrixWorkspace_sptr masked = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_masked_ws");
    auto &outX = masked->x(0);
    auto &outY = masked->y(0);

    const MatrixWorkspace::MaskList &mask = masked->maskedBins(0);

    // turn the mask list into an array like the Y values
    MantidVec weights(outY.size(), 0);
    for (auto it : mask) {
      weights[it.first] = it.second;
    }

    // the degree of masking must be the same as the reduction in the y-value,
    // for distributions, this is the easy case
    for (size_t i = 0; i < outY.size(); ++i) {
      size_t inBin = std::lower_bound(inX.begin(), inX.end(), outX[i]) - inX.begin();
      if (inBin < inX.size() - 2) {
        TS_ASSERT_DELTA(outY[i] / inY[inBin], 1 - weights[i], 0.000001);
      }
    }
    // the above formula tests the criterian that must be true for masking,
    // however we need some more specific tests incase outY.empty() or something
    TS_ASSERT_DELTA(outY[1], 0.0, 0.000001)
    TS_ASSERT_DELTA(outY[2], 1.0, 0.000001)
    TS_ASSERT_DELTA(outY[3], 6.0, 0.000001)
    TS_ASSERT_DELTA(weights[2], 1 - (0.25 / 3.0), 0.000001)

    AnalysisDataService::Instance().remove("test_Rebin_masked_ws");
    AnalysisDataService::Instance().remove("test_Rebin_unmasked");
    AnalysisDataService::Instance().remove("test_Rebin_mask_raw");
  }

  void test_FullBinsOnly_Fixed() {
    std::vector<double> xExpected = {0.5, 2.5, 4.5, 6.5};
    std::vector<double> yExpected(3, 8.0);
    std::string params = "2.0";
    do_test_FullBinsOnly(params, yExpected, xExpected);
  }

  void test_FullBinsOnly_Variable() {
    std::vector<double> xExpected = {0.5, 1.5, 2.5, 3.2, 3.9, 4.6, 6.6};
    std::vector<double> yExpected = {4.0, 4.0, 2.8, 2.8, 2.8, 8.0};
    std::string params = "0.5, 1.0, 3.1, 0.7, 5.0, 2.0, 7.25";
    do_test_FullBinsOnly(params, yExpected, xExpected);
  }

  void test_reverseLogSimple() {
    // Test UseReverseLogarithmic alone
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, -1, 37");
    rebin.setProperty("UseReverseLogarithmic", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 6);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 22, 1e-5);
    TS_ASSERT_DELTA(outX[2], 30, 1e-5);
    TS_ASSERT_DELTA(outX[3], 34, 1e-5);
    TS_ASSERT_DELTA(outX[4], 36, 1e-5);
    TS_ASSERT_DELTA(outX[5], 37, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_reverseLogDiffStep() {
    // Test UseReverseLog with a different step than the usual -1
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, -2, 42");
    rebin.setProperty("UseReverseLogarithmic", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 4);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 34, 1e-5);
    TS_ASSERT_DELTA(outX[2], 40, 1e-5);
    TS_ASSERT_DELTA(outX[3], 42, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_reverseLogEdgeCase() {
    // Check the case where the parameters given are so that the edges of the bins fall perfectly, and so no padding is
    // needed
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, -1, 16");
    rebin.setProperty("UseReverseLogarithmic", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 5);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5)
    TS_ASSERT_DELTA(outX[1], 9, 1e-5);
    TS_ASSERT_DELTA(outX[2], 13, 1e-5);
    TS_ASSERT_DELTA(outX[3], 15, 1e-5);
    TS_ASSERT_DELTA(outX[4], 16, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_reverseLogAgainst() {
    // Test UseReverseLogarithmic with a linear spacing before it
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, 2, 5, -1, 100");
    rebin.setProperty("UseReverseLogarithmic", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 7); // 2 lin + 4 log
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 3, 1e-5);
    TS_ASSERT_DELTA(outX[2], 5, 1e-5);
    TS_ASSERT_DELTA(outX[3], 65, 1e-5);
    TS_ASSERT_DELTA(outX[4], 85, 1e-5);
    TS_ASSERT_DELTA(outX[5], 95, 1e-5);
    TS_ASSERT_DELTA(outX[6], 100, 1e-5);

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_reverseLogBetween() {
    // Test UseReverseLogarithmic between 2 linear binnings

    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, 2, 5, -1, 100, 2, 110");
    rebin.setProperty("UseReverseLogarithmic", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 12);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 3, 1e-5);
    TS_ASSERT_DELTA(outX[2], 5, 1e-5);
    TS_ASSERT_DELTA(outX[3], 65, 1e-5);
    TS_ASSERT_DELTA(outX[4], 85, 1e-5);
    TS_ASSERT_DELTA(outX[5], 95, 1e-5);
    TS_ASSERT_DELTA(outX[6], 100, 1e-5);
    TS_ASSERT_DELTA(outX[7], 102, 1e-5);
    TS_ASSERT_DELTA(outX[11], 110, 1e-5);

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_reverseLogFullBinsOnly() {
    // Test UseReverseLogarithmic with the FullBinsOnly option checked. It should not change anything from the non
    // checked version.
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, -1, 37");
    rebin.setProperty("UseReverseLogarithmic", true);
    rebin.setProperty("FullBinsOnly", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 6);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 22, 1e-5);
    TS_ASSERT_DELTA(outX[2], 30, 1e-5);
    TS_ASSERT_DELTA(outX[3], 34, 1e-5);
    TS_ASSERT_DELTA(outX[4], 36, 1e-5);
    TS_ASSERT_DELTA(outX[5], 37, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_reverseLogIncompleteFirstBin() {
    // Test UseReverseLogarithmic with a first bin that is incomplete, but still bigger than the next one so not merged.
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, -1, 60");
    rebin.setProperty("UseReverseLogarithmic", true);
    rebin.setProperty("FullBinsOnly", true);
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 7);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 29, 1e-5);
    TS_ASSERT_DELTA(outX[2], 45, 1e-5);
    TS_ASSERT_DELTA(outX[3], 53, 1e-5);
    TS_ASSERT_DELTA(outX[4], 57, 1e-5);
    TS_ASSERT_DELTA(outX[5], 59, 1e-5);
    TS_ASSERT_DELTA(outX[6], 60, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_inversePowerSquareRoot() {
    // Test InversePower in a simple case of square root sum
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, 1, 10");
    rebin.setPropertyValue("Power", "0.5");
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 28);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 2, 1e-5);
    TS_ASSERT_DELTA(outX[2], 2.707106781, 1e-5);
    TS_ASSERT_DELTA(outX[3], 3.28445705, 1e-5);
    TS_ASSERT_DELTA(outX[27], 10, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_inversePowerHarmonic() {
    // Test InversePower in a simple case of harmonic serie
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, 1, 5");
    rebin.setPropertyValue("Power", "1");
    TS_ASSERT_THROWS_NOTHING(rebin.execute());

    MatrixWorkspace_sptr out = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_Rebin_revLog");
    auto &outX = out->x(0);

    TS_ASSERT_EQUALS(outX.size(), 31);
    TS_ASSERT_DELTA(outX[0], 1, 1e-5);
    TS_ASSERT_DELTA(outX[1], 2, 1e-5);
    TS_ASSERT_DELTA(outX[2], 2.5, 1e-5);
    TS_ASSERT_DELTA(outX[3], 2.8333333, 1e-5);
    TS_ASSERT_DELTA(outX[30], 5, 1e-5);

    TS_ASSERT(checkBinWidthMonotonic(out, Monotonic::DECREASE, true));

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_inversePowerValidateHarmonic() {
    // Test that the validator which forbid creating more than 10000 bins works in a harmonic series case
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, 1, 100");
    rebin.setPropertyValue("Power", "1");
    TS_ASSERT_THROWS(rebin.execute(), const std::runtime_error &);
    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_inversePowerValidateInverseSquareRoot() {
    // Test that the validator which forbid breating more than 10000 bins works in an inverse square root case
    // We test both because they rely on different formula to compute the expected number of bins.
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1, 1, 1000");
    rebin.setPropertyValue("Power", "0.5");
    TS_ASSERT_THROWS(rebin.execute(), const std::runtime_error &);
    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_inversePowerBinningMode() {
    // Test that if the mode is set to power mode, will produce power binning
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_in_Rebin", test_1D);

    // run once the default way
    Rebin rebin_once;
    rebin_once.initialize();
    rebin_once.setPropertyValue("InputWorkspace", "test_in_Rebin");
    rebin_once.setPropertyValue("OutputWorkspace", "test_out_Rebin_once");
    rebin_once.setPropertyValue("Params", "1, 1, 10");
    rebin_once.setPropertyValue("Power", "0.5");
    TS_ASSERT_THROWS_NOTHING(rebin_once.execute());

    MatrixWorkspace_sptr out_once = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_Rebin_once");
    auto &outX_once = out_once->x(0);

    // run again by setting the binning mode
    Rebin rebin_again;
    rebin_again.initialize();
    rebin_again.setPropertyValue("InputWorkspace", "test_in_Rebin");
    rebin_again.setPropertyValue("OutputWorkspace", "test_out_Rebin_again");
    rebin_again.setPropertyValue("Params", "1, -1, 10"); // set step to negative, just to confuse it
    rebin_again.setPropertyValue("Power", "0.5");
    rebin_again.setPropertyValue("BinningMode", "Power");
    TS_ASSERT_THROWS_NOTHING(rebin_again.execute());

    // the algo validation will set this to be compatible with power binning
    std::vector<double> rbParams = rebin_again.getProperty("Params");
    TS_ASSERT(rbParams[1] > 0);

    MatrixWorkspace_sptr out_again =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_Rebin_again");
    auto &outX_again = out_again->x(0);

    TS_ASSERT_EQUALS(outX_once.size(), outX_again.size());
    for (size_t i = 0; i < outX_once.size(); i++) {
      TS_ASSERT_EQUALS(outX_once[i], outX_again[i]);
    }

    TS_ASSERT(checkBinWidthMonotonic(out_once, Monotonic::DECREASE));
    TS_ASSERT(checkBinWidthMonotonic(out_again, Monotonic::DECREASE));

    AnalysisDataService::Instance().remove("test_in_Rebin");
    AnalysisDataService::Instance().remove("test_out_Rebin_once");
    AnalysisDataService::Instance().remove("test_out_Rebin_again");
  }

  void test_inversePowerFailure() {
    // Test InversePower will fail if binning mode set to Power but no power given
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_Rebin_revLog", test_1D);

    Rebin rebin;
    rebin.initialize();
    rebin.setPropertyValue("InputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("OutputWorkspace", "test_Rebin_revLog");
    rebin.setPropertyValue("Params", "1.2, 1.2, 12.22");
    rebin.setPropertyValue("BinningMode", "Power");
    TS_ASSERT_THROWS_ANYTHING(rebin.execute());

    AnalysisDataService::Instance().remove("test_Rebin_revLog");
  }

  void test_inverseNoPowerWithBin() {
    // Test that if the mode is set to linear, will NOT produce power binning,
    // even if the power is set to nonzero
    Workspace2D_sptr test_1D = Create1DWorkspace(51);
    test_1D->setDistribution(false);
    AnalysisDataService::Instance().add("test_in_Rebin", test_1D);

    // run once as normal linear binning
    Rebin rebin_once;
    rebin_once.initialize();
    rebin_once.setPropertyValue("InputWorkspace", "test_in_Rebin");
    rebin_once.setPropertyValue("OutputWorkspace", "test_out_Rebin_once");
    rebin_once.setPropertyValue("Params", "1.1, 1.1, 11.1");
    rebin_once.setPropertyValue("Power", "0.0");
    TS_ASSERT_THROWS_NOTHING(rebin_once.execute());

    MatrixWorkspace_sptr out_once = AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_Rebin_once");
    auto &outX_once = out_once->x(0);

    // run again by setting both the power and the binning mode to linear
    Rebin rebin_again;
    rebin_again.initialize();
    rebin_again.setPropertyValue("InputWorkspace", "test_in_Rebin");
    rebin_again.setPropertyValue("OutputWorkspace", "test_out_Rebin_again");
    rebin_again.setPropertyValue("Params", "1.1, 1.1, 11.1"); // set step to negative, just to confuse it
    rebin_again.setPropertyValue("Power", "0.5");
    rebin_again.setPropertyValue("BinningMode", "Linear");
    TS_ASSERT_THROWS_NOTHING(rebin_again.execute());

    // the algo validation will remove the power if Linear binning is set
    TS_ASSERT(double(rebin_again.getProperty("Power")) == 0.0);

    MatrixWorkspace_sptr out_again =
        AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>("test_out_Rebin_again");
    auto &outX_again = out_again->x(0);

    TS_ASSERT_EQUALS(outX_once.size(), outX_again.size());
    for (size_t i = 0; i < outX_once.size(); i++) {
      TS_ASSERT_EQUALS(outX_once[i], outX_again[i]);
    }

    AnalysisDataService::Instance().remove("test_in_Rebin");
    AnalysisDataService::Instance().remove("test_out_Rebin_once");
    AnalysisDataService::Instance().remove("test_out_Rebin_again");
  }

private:
  Workspace2D_sptr Create1DWorkspace(int size) {
    auto retVal = createWorkspace<Workspace2D>(1, size, size - 1);
    double j = 0.5;
    for (int i = 0; i < size; i++) {
      retVal->dataX(0)[i] = j;
      j += 0.75;
    }
    retVal->setCounts(0, size - 1, 3.0);
    retVal->setCountVariances(0, size - 1, 3.0);
    return retVal;
  }

  Workspace2D_sptr Create2DWorkspace(int xlen, int ylen) {
    BinEdges x1(xlen, HistogramData::LinearGenerator(0.5, 0.75));
    Counts y1(xlen - 1, 3.0);
    CountStandardDeviations e1(xlen - 1, sqrt(3.0));

    auto retVal = createWorkspace<Workspace2D>(ylen, xlen, xlen - 1);

    for (int i = 0; i < ylen; i++) {
      retVal->setBinEdges(i, x1);
      retVal->setCounts(i, y1);
      retVal->setCountStandardDeviations(i, e1);
    }

    return retVal;
  }

  void maskFirstBins(const std::string &in, const std::string &out, double maskBinsTo) {
    MaskBins mask;
    mask.initialize();
    mask.setPropertyValue("InputWorkspace", in);
    mask.setPropertyValue("OutputWorkspace", out);
    mask.setProperty("XMin", 0.0);
    mask.setProperty("XMax", maskBinsTo);
    mask.execute();
  }

  void do_test_FullBinsOnly(const std::string &params, const std::vector<double> &yExpected,
                            const std::vector<double> &xExpected) {
    ScopedWorkspace inWsEntry(Create1DWorkspace(10));
    ScopedWorkspace outWsEntry;

    try {
      Rebin rebin;
      rebin.initialize();
      rebin.setPropertyValue("InputWorkspace", inWsEntry.name());
      rebin.setPropertyValue("OutputWorkspace", outWsEntry.name());
      rebin.setPropertyValue("Params", params);
      rebin.setProperty("FullBinsOnly", true);
      rebin.execute();
    } catch (std::runtime_error &e) {
      TS_FAIL(e.what());
      return;
    }

    auto ws = std::dynamic_pointer_cast<MatrixWorkspace>(outWsEntry.retrieve());

    if (!ws) {
      TS_FAIL("Unable to retrieve result workspace");
      return; // Nothing else to check
    }

    auto &xValues = ws->x(0);
    TS_ASSERT_DELTA(xValues.rawData(), xExpected, 0.001);

    auto &yValues = ws->y(0);
    TS_ASSERT_DELTA(yValues.rawData(), yExpected, 0.001);
  }

  enum class Monotonic : bool { INCREASE = false, DECREASE = true };
  bool checkBinWidthMonotonic(MatrixWorkspace_sptr ws, Monotonic monotonic = Monotonic::INCREASE,
                              bool ignoreLastBin = false) {
    size_t binEdgesTotal = ws->blocksize();
    if (ignoreLastBin)
      binEdgesTotal--;
    auto binEdges = ws->binEdges(0);
    double lastBinSize = binEdges[1] - binEdges[0];
    bool allMonotonic = true;
    for (size_t i = 1; i < binEdgesTotal && allMonotonic; ++i) {
      double currentBinSize = binEdges[i + 1] - binEdges[i];

      switch (monotonic) {
      case Monotonic::INCREASE:
        allMonotonic &= lastBinSize < currentBinSize;
        break;
      case Monotonic::DECREASE:
        allMonotonic &= lastBinSize > currentBinSize;
        break;
      }
      lastBinSize = currentBinSize;
    }
    return allMonotonic;
  }
};

class RebinTestPerformance : public CxxTest::TestSuite {
public:
  // This pair of boilerplate methods prevent the suite being created statically
  // This means the constructor isn't called when running other tests
  static RebinTestPerformance *createSuite() { return new RebinTestPerformance(); }
  static void destroySuite(RebinTestPerformance *suite) { delete suite; }

  RebinTestPerformance() { ws = WorkspaceCreationHelper::create2DWorkspaceBinned(5000, 20000); }

  void test_rebin() {
    Rebin rebin;
    rebin.initialize();
    rebin.setProperty("InputWorkspace", ws);
    rebin.setPropertyValue("OutputWorkspace", "out");
    rebin.setPropertyValue("Params", "50,1.77,18801");
    TS_ASSERT(rebin.execute());
  }

private:
  MatrixWorkspace_sptr ws;
};
