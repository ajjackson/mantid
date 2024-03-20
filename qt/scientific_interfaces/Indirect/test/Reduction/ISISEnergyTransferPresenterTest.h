// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2019 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include <cxxtest/TestSuite.h>
#include <gmock/gmock.h>

#include "../../../Inelastic/test/Common/MockObjects.h"
#include "../Common/MockObjects.h"
#include "MockObjects.h"
#include "Reduction/ISISEnergyTransferPresenter.h"

#include "MantidAPI/AlgorithmProperties.h"
#include "MantidFrameworkTestHelpers/IndirectFitDataCreationHelper.h"
#include "MantidKernel/WarningSuppressions.h"

using namespace MantidQt::CustomInterfaces;
using namespace testing;

namespace {

std::unique_ptr<Mantid::API::AlgorithmRuntimeProps> defaultGroupingProps() {
  auto properties = std::make_unique<Mantid::API::AlgorithmRuntimeProps>();
  Mantid::API::AlgorithmProperties::update("GroupingMethod", std::string("IPF"), *properties);
  return properties;
}

IETSaveData createSaveData() { return IETSaveData(true, true, true, true, true); }

} // namespace

class ISISEnergyTransferPresenterTest : public CxxTest::TestSuite {
public:
  static ISISEnergyTransferPresenterTest *createSuite() { return new ISISEnergyTransferPresenterTest(); }
  static void destroySuite(ISISEnergyTransferPresenterTest *suite) { delete suite; }

  void setUp() override {
    auto model = std::make_unique<NiceMock<MockIETModel>>();

    m_outputOptionsView = std::make_unique<NiceMock<MockOutputPlotOptionsView>>();
    m_instrumentConfig = std::make_unique<NiceMock<MockInstrumentConfig>>();

    m_view = std::make_unique<NiceMock<MockIETView>>();
    ON_CALL(*m_view, getPlotOptionsView()).WillByDefault(Return(m_outputOptionsView.get()));

    m_model = model.get();
    m_idrUI = std::make_unique<NiceMock<MockIndirectDataReduction>>();
    ON_CALL(*m_idrUI, getInstrumentConfiguration()).WillByDefault(Return(m_instrumentConfig.get()));

    m_presenter = std::make_unique<IETPresenter>(m_idrUI.get(), m_view.get(), std::move(model));
  }

  void tearDown() override {
    TS_ASSERT(Mock::VerifyAndClearExpectations(&m_outputOptionsView));
    TS_ASSERT(Mock::VerifyAndClearExpectations(&m_instrumentConfig));
    TS_ASSERT(Mock::VerifyAndClearExpectations(&m_view));
    TS_ASSERT(Mock::VerifyAndClearExpectations(m_model));
    TS_ASSERT(Mock::VerifyAndClearExpectations(&m_idrUI));

    m_presenter.reset();
    m_idrUI.reset();
    m_view.reset();
    m_outputOptionsView.reset();
    m_instrumentConfig.reset();
  }

  void test_notifySaveClicked_will_not_save_the_workspace_if_its_not_in_the_ADS() {
    std::vector<std::string> names{"NotInADS"};

    ON_CALL(*m_view, getSaveData()).WillByDefault(Return(createSaveData()));
    ON_CALL(*m_model, outputWorkspaceNames()).WillByDefault(Return(names));

    // Expect no call because the workspace is not in the ADS
    EXPECT_CALL(*m_model, saveWorkspace(_, _)).Times(0);

    m_presenter->notifySaveClicked();
  }

  void test_notifySaveClicked_will_save_the_workspace_if_its_not_in_the_ADS() {
    std::vector<std::string> names{"InADS"};
    auto const workspace = Mantid::IndirectFitDataCreationHelper::createWorkspace(4, 5);
    AnalysisDataService::Instance().addOrReplace(names[0], workspace);

    ON_CALL(*m_view, getSaveData()).WillByDefault(Return(createSaveData()));
    ON_CALL(*m_model, outputWorkspaceNames()).WillByDefault(Return(names));

    EXPECT_CALL(*m_model, saveWorkspace(names[0], _)).Times(1);

    m_presenter->notifySaveClicked();
  }

  void test_notifyRunFinished_sets_run_text_to_invalid_if_the_run_files_are_not_valid() {
    ON_CALL(*m_view, isRunFilesValid()).WillByDefault(Return(false));

    EXPECT_CALL(*m_view, setRunButtonText("Invalid Run(s)")).Times(1);
    EXPECT_CALL(*m_view, setRunFilesEnabled(true)).Times(1);

    m_presenter->notifyRunFinished();
  }

  void test_notifyRunFinished_sets_the_run_text_when_the_run_files_are_valid() {
    std::string const filename = "filename.nxs";
    double const detailedBalance = 1.1;

    ON_CALL(*m_view, isRunFilesValid()).WillByDefault(Return(true));

    ON_CALL(*m_view, getFirstFilename()).WillByDefault(Return(filename));
    ON_CALL(*m_model, loadDetailedBalance(filename)).WillByDefault(Return(detailedBalance));

    EXPECT_CALL(*m_view, setDetailedBalance(detailedBalance)).Times(1);
    EXPECT_CALL(*m_view, setRunButtonText("Run")).Times(1);
    EXPECT_CALL(*m_view, setRunFilesEnabled(true)).Times(1);

    m_presenter->notifyRunFinished();
  }

private:
  std::unique_ptr<IETPresenter> m_presenter;

  std::unique_ptr<NiceMock<MockIETView>> m_view;
  NiceMock<MockIETModel> *m_model;
  std::unique_ptr<NiceMock<MockIndirectDataReduction>> m_idrUI;

  std::unique_ptr<NiceMock<MockOutputPlotOptionsView>> m_outputOptionsView;
  std::unique_ptr<NiceMock<MockInstrumentConfig>> m_instrumentConfig;
};