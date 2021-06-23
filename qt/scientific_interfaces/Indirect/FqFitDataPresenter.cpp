// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "FqFitDataPresenter.h"
#include "FqFitDataTablePresenter.h"
#include "IDAFunctionParameterEstimation.h"

#include "MantidQtWidgets/Common/SignalBlocker.h"

namespace {
auto &ads_instance = Mantid::API::AnalysisDataService::Instance();
}

namespace MantidQt {
namespace CustomInterfaces {
namespace IDA {

FqFitDataPresenter::FqFitDataPresenter(FqFitModel *model, IIndirectFitDataView *view,
                                       IFQFitObserver *SingleFunctionTemplateBrowser)
    : IndirectFitDataPresenter(model, view, std::make_unique<FqFitDataTablePresenter>(model, view->getDataTable())),
      m_activeParameterType("Width"), m_dataIndex(TableDatasetIndex{0}), m_fqFitModel(model) {

  connect(this, SIGNAL(requestedAddWorkspaceDialog()), this, SLOT(updateActiveDataIndex()));

  m_notifier = Notifier<IFQFitObserver>();
  m_notifier.subscribe(SingleFunctionTemplateBrowser);
}

void FqFitDataPresenter::setActiveParameterType(const std::string &type) { m_activeParameterType = type; }

void FqFitDataPresenter::updateActiveDataIndex() { m_dataIndex = m_fqFitModel->getNumberOfWorkspaces(); }

void FqFitDataPresenter::updateActiveDataIndex(int index) { m_dataIndex = index; }

void FqFitDataPresenter::setDialogParameterNames(FqFitAddWorkspaceDialog *dialog, const std::string &workspaceName) {
  FqFitParameters parameters;
  try {
    auto workspace = ads_instance.retrieveWS<MatrixWorkspace>(workspaceName);
    parameters = m_fqFitModel->createFqFitParameters(workspace.get());
    dialog->enableParameterSelection();
  } catch (const std::invalid_argument &) {
    dialog->disableParameterSelection();
  }
  updateParameterTypes(dialog, parameters);
  updateParameterOptions(dialog, parameters);
}

void FqFitDataPresenter::dialogParameterTypeUpdated(FqFitAddWorkspaceDialog *dialog, const std::string &type) {
  const auto workspaceName = dialog->workspaceName();
  auto workspace = ads_instance.retrieveWS<MatrixWorkspace>(workspaceName);
  const auto parameter = m_fqFitModel->createFqFitParameters(workspace.get());
  setActiveParameterType(type);
  updateParameterOptions(dialog, parameter);
}

void FqFitDataPresenter::updateParameterOptions(FqFitAddWorkspaceDialog *dialog, FqFitParameters parameter) {
  setDataIndexToCurrentWorkspace(dialog);
  setActiveParameterType(dialog->parameterType());
  if (m_activeParameterType == "Width")
    dialog->setParameterNames(parameter.widths);
  else if (m_activeParameterType == "EISF")
    dialog->setParameterNames(parameter.eisf);
  else
    dialog->setParameterNames({});
}

void FqFitDataPresenter::updateParameterTypes(FqFitAddWorkspaceDialog *dialog, FqFitParameters &parameters) {
  setDataIndexToCurrentWorkspace(dialog);
  dialog->setParameterTypes(getParameterTypes(parameters));
}

std::vector<std::string> FqFitDataPresenter::getParameterTypes(FqFitParameters &parameters) const {
  std::vector<std::string> types;
  if (!parameters.widths.empty())
    types.emplace_back("Width");
  if (!parameters.eisf.empty())
    types.emplace_back("EISF");
  return types;
}

void FqFitDataPresenter::addWorkspace(IndirectFittingModel *model, const std::string &name) {
  if (model->getNumberOfWorkspaces() > m_dataIndex)
    model->removeWorkspace(m_dataIndex);
  model->addWorkspace(name);
}

void FqFitDataPresenter::addDataToModel(IAddWorkspaceDialog const *dialog) {
  if (const auto fqFitDialog = dynamic_cast<FqFitAddWorkspaceDialog const *>(dialog)) {
    m_fqFitModel->addWorkspace(fqFitDialog->workspaceName(), fqFitDialog->parameterNameIndex());
    setDataIndexToCurrentWorkspace(fqFitDialog);
    // here we can say that we are in multiple mode so we can append the spectra
    // to the current one and then setspectra
    setModelSpectrum(fqFitDialog->parameterNameIndex());
    updateActiveDataIndex();
  }
}

void FqFitDataPresenter::setModelSpectrum(int index) {
  if (index < 0)
    throw std::runtime_error("No valid parameter was selected.");
  else if (m_activeParameterType == "Width")
    m_fqFitModel->setActiveWidth(static_cast<std::size_t>(index), m_dataIndex, false);
  else
    m_fqFitModel->setActiveEISF(static_cast<std::size_t>(index), m_dataIndex, false);
}

void FqFitDataPresenter::setDataIndexToCurrentWorkspace(IAddWorkspaceDialog const *dialog) {
  //  update active data index with correct index based on the workspace name
  //  and the vector in m_fitDataModel which is in the base class
  //  indirectFittingModel get table workspace index
  const auto wsName = dialog->workspaceName().append("_HWHM");
  // This a vector of workspace names currently loaded
  auto wsVector = m_fqFitModel->getFitDataModel()->getWorkspaceNames();
  // this is an iterator pointing to the current wsName in wsVector
  auto wsIt = std::find(wsVector.begin(), wsVector.end(), wsName);
  // this is the index of the workspace.
  int index = int(std::distance(wsVector.begin(), wsIt));
  updateActiveDataIndex(index);
}

void FqFitDataPresenter::closeDialog() {
  if (m_fqFitModel->getNumberOfWorkspaces() > m_dataIndex)
    m_fqFitModel->removeWorkspace(m_dataIndex);
  IndirectFitDataPresenter::closeDialog();
}

std::unique_ptr<IAddWorkspaceDialog> FqFitDataPresenter::getAddWorkspaceDialog(QWidget *parent) const {
  auto dialog = std::make_unique<FqFitAddWorkspaceDialog>(parent);
  connect(dialog.get(), SIGNAL(workspaceChanged(FqFitAddWorkspaceDialog *, const std::string &)), this,
          SLOT(setDialogParameterNames(FqFitAddWorkspaceDialog *, const std::string &)));
  connect(dialog.get(), SIGNAL(parameterTypeChanged(FqFitAddWorkspaceDialog *, const std::string &)), this,
          SLOT(dialogParameterTypeUpdated(FqFitAddWorkspaceDialog *, const std::string &)));
  return dialog;
}

} // namespace IDA
} // namespace CustomInterfaces
} // namespace MantidQt
