// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "Analysis/ParameterEstimation.h"
#include "DllConfig.h"
#include "ITemplatePresenter.h"
#include "IqtFunctionModel.h"

class QtProperty;

namespace MantidQt {
namespace MantidWidgets {
class EditLocalParameterDialog;
}
namespace CustomInterfaces {
namespace IDA {

class IqtTemplateBrowser;

/**
 * Class FunctionTemplateBrowser implements QtPropertyBrowser to display
 * and set properties that can be used to generate a fit function.
 *
 */
class MANTIDQT_INELASTIC_DLL IqtTemplatePresenter : public ITemplatePresenter {
public:
  explicit IqtTemplatePresenter(IqtTemplateBrowser *view, std::unique_ptr<IqtFunctionModel> functionModel);
  FunctionTemplateBrowser *browser() override { return reinterpret_cast<FunctionTemplateBrowser *>(m_view); }

  void setNumberOfExponentials(int) override;
  void setStretchExponential(bool) override;
  void setBackground(std::string const &name) override;

  void setNumberOfDatasets(int) override;
  int getNumberOfDatasets() const override;
  int getCurrentDataset() override;

  void setFunction(std::string const &funStr) override;
  IFunction_sptr getGlobalFunction() const override;
  IFunction_sptr getFunction() const override;

  std::vector<std::string> getGlobalParameters() const override;
  std::vector<std::string> getLocalParameters() const override;
  void setGlobalParameters(std::vector<std::string> const &globals) override;
  void setGlobal(std::string const &parameterName, bool on) override;

  void updateMultiDatasetParameters(const IFunction &fun) override;
  void updateMultiDatasetParameters(const ITableWorkspace &paramTable) override;
  void updateParameters(const IFunction &fun) override;

  void setCurrentDataset(int i) override;
  void setDatasets(const QList<FunctionModelDataset> &datasets) override;

  void setViewParameterDescriptions();
  void tieIntensities(bool on) override;
  bool canTieIntensities() const override;

  EstimationDataSelector getEstimationDataSelector() const override;
  void updateParameterEstimationData(DataForParameterEstimationCollection &&data) override;
  void estimateFunctionParameters() override;

  void setErrorsEnabled(bool enabled) override;

  void setBackgroundA0(double value) override;

  void handleEditLocalParameter(std::string const &parameterName) override;
  void handleParameterValueChanged(std::string const &parameterName, double value) override;
  void handleEditLocalParameterFinished(std::string const &parameterName, QList<double> const &values,
                                        QList<bool> const &fixes, QStringList const &ties,
                                        QStringList const &constraints) override;

private:
  void updateViewParameters();
  QStringList getDatasetNames() const;
  QStringList getDatasetDomainNames() const;
  double getLocalParameterValue(std::string const &parameterName, int i) const;
  bool isLocalParameterFixed(std::string const &parameterName, int i) const;
  std::string getLocalParameterTie(std::string const &parameterName, int i) const;
  std::string getLocalParameterConstraint(std::string const &parameterName, int i) const;
  void setLocalParameterValue(std::string const &parameterName, int i, double value);
  void setLocalParameterFixed(std::string const &parameterName, int i, bool fixed);
  void setLocalParameterTie(std::string const &parameterName, int i, std::string const &tie);
  void updateViewParameterNames();
  void updateView();
  IqtTemplateBrowser *m_view;
  std::unique_ptr<IqtFunctionModel> m_model;
};

} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt
