// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidAlgorithms/ConvertToPointData.h"
#include <cxxtest/TestSuite.h>

#include "MantidAPI/NumericAxis.h"
#include "MantidFrameworkTestHelpers/WorkspaceCreationHelper.h"
#include "MantidKernel/Unit.h"
#include "MantidKernel/UnitFactory.h"
using Mantid::Algorithms::ConvertToPointData;
using Mantid::API::IAlgorithm_sptr;
using Mantid::API::MatrixWorkspace;
using Mantid::API::MatrixWorkspace_sptr;
using Mantid::DataObjects::Workspace2D_sptr;
using Mantid::HistogramData::BinEdges;
using Mantid::HistogramData::Counts;
using Mantid::HistogramData::Histogram;

class ConvertToPointDataTest : public CxxTest::TestSuite {

public:
  void tearDown() override { Mantid::API::AnalysisDataService::Instance().clear(); }

  void test_That_The_Algorithm_Has_Two_Properties() {
    ConvertToPointData alg;
    TS_ASSERT_THROWS_NOTHING(alg.initialize());
    TS_ASSERT_EQUALS(alg.getProperties().size(), 2);
  }

  void test_That_Output_Is_The_Same_As_Input_If_Input_Contains_Point_Data() {
    // False indicates a non histogram workspace
    Workspace2D_sptr testWS = WorkspaceCreationHelper::create2DWorkspace123(5, 10, false);

    MatrixWorkspace_sptr outputWS = runAlgorithm(testWS);
    TS_ASSERT(outputWS);
    if (!outputWS)
      return;

    // Check that the algorithm just pointed the output data at the input
    TS_ASSERT_EQUALS(&(*testWS), &(*outputWS));
  }

  void test_A_Uniformly_Binned_Histogram_Is_Transformed_Correctly() {

    // Creates a workspace with 2 spectra, 10 bins with bin width 1.0 starting
    // from 0.0
    const int numBins(10);
    const int numSpectra(2);
    Workspace2D_sptr testWS = WorkspaceCreationHelper::create2DWorkspaceBinned(numSpectra, numBins);
    TS_ASSERT_EQUALS(testWS->isHistogramData(), true);
    // add a new vertical axis
    auto verticalAxis = std::make_unique<Mantid::API::NumericAxis>(numSpectra + 1);

    for (int i = 0; i < numSpectra + 1; i++) {
      verticalAxis->setValue(i, 2 * i);
    }
    verticalAxis->unit() = Mantid::Kernel::UnitFactory::Instance().create("MomentumTransfer");
    verticalAxis->title() = "|Q|";
    testWS->replaceAxis(1, std::move(verticalAxis));
    MatrixWorkspace_sptr outputWS = runAlgorithm(testWS);

    TS_ASSERT(outputWS);
    if (!outputWS)
      return;

    TS_ASSERT_EQUALS(outputWS->isHistogramData(), false);
    for (int i = 0; i < numSpectra; ++i) {
      const Mantid::MantidVec &yValues = outputWS->readY(i);
      const Mantid::MantidVec &xValues = outputWS->readX(i);
      const Mantid::MantidVec &eValues = outputWS->readE(i);

      // The X size should be now equal to the number of bins
      TS_ASSERT_EQUALS(xValues.size(), numBins);
      // The y and e values sizes be unchanged
      TS_ASSERT_EQUALS(yValues.size(), numBins);
      TS_ASSERT_EQUALS(eValues.size(), numBins);

      for (int j = 0; j < numBins; ++j) {
        // Now the data. Y and E unchanged
        TS_ASSERT_EQUALS(yValues[j], 2.0);
        TS_ASSERT_EQUALS(eValues[j], M_SQRT2);
        // X data originally was 0->10 in steps of 1. Now it should be the
        // centre of each bin which is
        // 1.0 away from the last centre
        const double expectedX = 0.5 + j * 1.0;
        TS_ASSERT_EQUALS(xValues[j], expectedX);
      }
    }
    // test the vertical axis
    TS_ASSERT_EQUALS(outputWS->getAxis(1)->length(), 3);
    TS_ASSERT_EQUALS(outputWS->getAxis(1)->unit()->unitID(), "MomentumTransfer");
    TS_ASSERT_EQUALS((*(outputWS->getAxis(1)))(0), 0);
    TS_ASSERT_EQUALS((*(outputWS->getAxis(1)))(1), 2);
    TS_ASSERT_EQUALS((*(outputWS->getAxis(1)))(2), 4);
  }

  void test_A_Non_Uniformly_Binned_Histogram_Is_Transformed_Correctly() {
    // Creates a workspace with 2 spectra, and the given bin structure
    double xBoundaries[11] = {0.0, 1.0, 3.0, 5.0, 6.0, 7.0, 10.0, 13.0, 16.0, 17.0, 17.5};
    const int numSpectra(2);
    Workspace2D_sptr testWS = WorkspaceCreationHelper::create2DWorkspaceNonUniformlyBinned(numSpectra, 11, xBoundaries);
    const size_t numBins = testWS->blocksize();
    TS_ASSERT_EQUALS(testWS->isHistogramData(), true);

    MatrixWorkspace_sptr outputWS = runAlgorithm(testWS);

    TS_ASSERT(outputWS);
    if (!outputWS)
      return;

    TS_ASSERT_EQUALS(outputWS->isHistogramData(), false);
    for (int i = 0; i < numSpectra; ++i) {
      const Mantid::MantidVec &yValues = outputWS->readY(i);
      const Mantid::MantidVec &xValues = outputWS->readX(i);
      const Mantid::MantidVec &eValues = outputWS->readE(i);

      // The X size should be now equal to the number of bins
      TS_ASSERT_EQUALS(xValues.size(), numBins);
      // The y and e values sizes be unchanged
      TS_ASSERT_EQUALS(yValues.size(), numBins);
      TS_ASSERT_EQUALS(eValues.size(), numBins);

      for (size_t j = 0; j < numBins; ++j) {
        // Now the data. Y and E unchanged
        TS_ASSERT_EQUALS(yValues[j], 2.0);
        TS_ASSERT_EQUALS(eValues[j], M_SQRT2);
        // X data originally was 0->10 in steps of 1. Now it should be the
        // centre of each bin which is
        // 1.0 away from the last centre
        const double expectedX = 0.5 * (xBoundaries[j] + xBoundaries[j + 1]);
        TS_ASSERT_EQUALS(xValues[j], expectedX);
      }
    }
  }

  void test_Dx_Data_Is_Handled_Correctly() {
    constexpr size_t numBins{11};
    double xBoundaries[numBins] = {0.0, 1.0, 3.0, 5.0, 6.0, 7.0, 10.0, 13.0, 16.0, 17.0, 17.5};
    constexpr int numSpectra{2};
    Workspace2D_sptr testWS =
        WorkspaceCreationHelper::create2DWorkspaceNonUniformlyBinned(numSpectra, numBins, xBoundaries, true);
    TS_ASSERT(testWS->isHistogramData())
    double xErrors[numBins - 1] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    MatrixWorkspace_sptr outputWS = runAlgorithm(testWS);
    TS_ASSERT(outputWS)
    TS_ASSERT(!outputWS->isHistogramData())
    for (size_t i = 0; i < outputWS->getNumberHistograms(); ++i) {
      TS_ASSERT(outputWS->hasDx(i))
      const auto &dx = outputWS->dx(i);
      TS_ASSERT_EQUALS(dx.size(), numBins - 1)
      for (size_t j = 0; j < dx.size(); ++j) {
        TS_ASSERT_DELTA(dx[j], xErrors[j], 1E-16);
      }
    }
  }

  void test_ragged() {
    // create ragged workspace
    Workspace2D_sptr raggedWS = WorkspaceCreationHelper::create2DWorkspace(2, 1);

    // create and replace histograms with ragged ones
    raggedWS->setHistogram(0, Histogram(BinEdges{100., 200., 300., 400.}, Counts{1., 2., 3.}));
    raggedWS->setHistogram(1, Histogram(BinEdges{200., 400., 600.}, Counts{4., 5.}));

    // quick check of the input workspace
    TS_ASSERT(raggedWS->isRaggedWorkspace());
    TS_ASSERT(raggedWS->isHistogramData())
    TS_ASSERT_EQUALS(raggedWS->getNumberHistograms(), 2);

    MatrixWorkspace_sptr outputWS = runAlgorithm(raggedWS);
    TS_ASSERT(outputWS)
    TS_ASSERT(!outputWS->isHistogramData())  // output is a points workspace
    TS_ASSERT(outputWS->isRaggedWorkspace()) // output is a ragged workspace
    TS_ASSERT_EQUALS(outputWS->getNumberHistograms(), 2);

    // check the data
    const Mantid::MantidVec &Y0 = outputWS->readY(0);
    const Mantid::MantidVec &X0 = outputWS->readX(0);
    const Mantid::MantidVec &Y1 = outputWS->readY(1);
    const Mantid::MantidVec &X1 = outputWS->readX(1);
    TS_ASSERT_EQUALS(Y0.size(), 3);
    TS_ASSERT_EQUALS(X0.size(), 3);
    TS_ASSERT_EQUALS(Y1.size(), 2);
    TS_ASSERT_EQUALS(X1.size(), 2);
    TS_ASSERT_EQUALS(X0[0], 150.);
    TS_ASSERT_EQUALS(X0[1], 250.);
    TS_ASSERT_EQUALS(X0[2], 350.);
    TS_ASSERT_EQUALS(X1[0], 300.);
    TS_ASSERT_EQUALS(X1[1], 500.);
    TS_ASSERT_EQUALS(Y0[0], 1.);
    TS_ASSERT_EQUALS(Y0[1], 2.);
    TS_ASSERT_EQUALS(Y0[2], 3.);
    TS_ASSERT_EQUALS(Y1[0], 4.);
    TS_ASSERT_EQUALS(Y1[1], 5.);
  }

private:
  MatrixWorkspace_sptr runAlgorithm(const Workspace2D_sptr &inputWS) {
    IAlgorithm_sptr alg(new ConvertToPointData());
    alg->initialize();
    alg->setRethrows(true);
    TS_ASSERT_THROWS_NOTHING(alg->setProperty<MatrixWorkspace_sptr>("InputWorkspace", inputWS));
    const std::string outputName = "__algOut";
    TS_ASSERT_THROWS_NOTHING(alg->setPropertyValue("OutputWorkspace", outputName));
    TS_ASSERT_THROWS_NOTHING(alg->execute());

    MatrixWorkspace_sptr outputWS =
        std::dynamic_pointer_cast<MatrixWorkspace>(Mantid::API::AnalysisDataService::Instance().retrieve(outputName));

    return outputWS;
  }
};

class ConvertToPointDataTestPerformance : public CxxTest::TestSuite {

public:
  // This pair of boilerplate methods prevent the suite being created statically
  // This means the constructor isn't called when running other tests
  static ConvertToPointDataTestPerformance *createSuite() { return new ConvertToPointDataTestPerformance(); }

  static void destroySuite(ConvertToPointDataTestPerformance *suite) { delete suite; }

  void setUp() override { inputWS = WorkspaceCreationHelper::create2DWorkspaceBinned(20000, 10000); }

  void tearDown() override { Mantid::API::AnalysisDataService::Instance().remove("output"); }

  void testPerformanceWS() {
    ConvertToPointData ctpd;
    ctpd.initialize();
    ctpd.setProperty("InputWorkspace", inputWS);
    ctpd.setProperty("OutputWorkspace", "output");
    ctpd.execute();
  }

private:
  Workspace2D_sptr inputWS;
};
