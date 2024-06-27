// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "FunctionQDataPresenter.h"
#include "FitTab.h"
#include "FitTabConstants.h"
#include "MantidAPI/TextAxis.h"
#include "ParameterEstimation.h"

#include "MantidAPI/AlgorithmManager.h"

namespace {
using namespace MantidQt::CustomInterfaces::Inelastic;

Mantid::Kernel::Logger logger("FunctionQDataPresenter");

struct ContainsOneOrMore {
  explicit ContainsOneOrMore(std::vector<std::string> &&substrings) : m_substrings(std::move(substrings)) {}

  bool operator()(const std::string &str) const {
    return std::any_of(m_substrings.cbegin(), m_substrings.cend(),
                       [&str](std::string const &substring) { return str.rfind(substring) != std::string::npos; });
  }

private:
  std::vector<std::string> m_substrings;
};

template <typename Predicate>
std::pair<std::vector<std::string>, std::vector<std::size_t>> findAxisLabels(TextAxis const *axis,
                                                                             Predicate const &predicate) {
  std::vector<std::string> labels;
  std::vector<std::size_t> spectra;

  for (auto i = 0u; i < axis->length(); ++i) {
    auto label = axis->label(i);
    if (predicate(label)) {
      labels.emplace_back(label);
      spectra.emplace_back(i);
    }
  }
  return std::make_pair(labels, spectra);
}

template <typename Predicate>
std::pair<std::vector<std::string>, std::vector<std::size_t>> findAxisLabels(MatrixWorkspace_sptr const &workspace,
                                                                             Predicate const &predicate) {
  auto axis = dynamic_cast<TextAxis *>(workspace->getAxis(1));
  if (axis)
    return findAxisLabels(axis, predicate);
  return std::make_pair(std::vector<std::string>(), std::vector<std::size_t>());
}

FunctionQParameters createFunctionQParameters(const MatrixWorkspace_sptr &workspace) {
  auto foundWidths = findAxisLabels(workspace, ContainsOneOrMore({".Width", ".FWHM"}));
  auto foundEISF = findAxisLabels(workspace, ContainsOneOrMore({".EISF"}));

  FunctionQParameters parameters;
  parameters.widths = foundWidths.first;
  parameters.widthSpectra = foundWidths.second;
  parameters.eisf = foundEISF.first;
  parameters.eisfSpectra = foundEISF.second;
  return parameters;
}

std::string createSpectra(const std::vector<std::size_t> &spectrum) {
  std::string spectra = "";
  for (const auto spec : spectrum) {
    spectra.append(std::to_string(spec) + ",");
  }
  return spectra;
}

std::string getHWHMName(const std::string &resultName) {
  auto position = resultName.rfind("_FWHM");
  if (position != std::string::npos)
    return resultName.substr(0, position) + "_HWHM" + resultName.substr(position + 5, resultName.size());
  return resultName + "_HWHM";
}

void deleteTemporaryWorkspaces(std::vector<std::string> const &workspaceNames) {
  auto deleter = AlgorithmManager::Instance().create("DeleteWorkspace");
  deleter->setLogging(false);
  for (const auto &name : workspaceNames) {
    deleter->setProperty("Workspace", name);
    deleter->execute();
  }
}

std::string scaleWorkspace(std::string const &inputName, std::string const &outputName, double factor) {
  auto scaleAlg = AlgorithmManager::Instance().create("Scale");
  scaleAlg->initialize();
  scaleAlg->setLogging(false);
  scaleAlg->setProperty("InputWorkspace", inputName);
  scaleAlg->setProperty("OutputWorkspace", outputName);
  scaleAlg->setProperty("Factor", factor);
  scaleAlg->execute();
  return outputName;
}

std::string extractSpectra(std::string const &inputName, int startIndex, int endIndex, std::string const &outputName) {
  auto extractAlg = AlgorithmManager::Instance().create("ExtractSpectra");
  extractAlg->initialize();
  extractAlg->setLogging(false);
  extractAlg->setProperty("InputWorkspace", inputName);
  extractAlg->setProperty("StartWorkspaceIndex", startIndex);
  extractAlg->setProperty("EndWorkspaceIndex", endIndex);
  extractAlg->setProperty("OutputWorkspace", outputName);
  extractAlg->execute();
  return outputName;
}

std::string extractSpectrum(const MatrixWorkspace_sptr &workspace, int index, std::string const &outputName) {
  return extractSpectra(workspace->getName(), index, index, outputName);
}

std::string extractHWHMSpectrum(const MatrixWorkspace_sptr &workspace, int index) {
  auto const scaledName = "__scaled_" + std::to_string(index);
  auto const extractedName = "__extracted_" + std::to_string(index);
  auto const outputName = scaleWorkspace(extractSpectrum(workspace, index, extractedName), scaledName, 0.5);
  deleteTemporaryWorkspaces({extractedName});
  return outputName;
}

std::string appendWorkspace(std::string const &lhsName, std::string const &rhsName, std::string const &outputName) {
  auto appendAlg = AlgorithmManager::Instance().create("AppendSpectra");
  appendAlg->initialize();
  appendAlg->setLogging(false);
  appendAlg->setProperty("InputWorkspace1", lhsName);
  appendAlg->setProperty("InputWorkspace2", rhsName);
  appendAlg->setProperty("OutputWorkspace", outputName);
  appendAlg->execute();
  return outputName;
}

MatrixWorkspace_sptr appendAll(std::vector<std::string> const &workspaces, std::string const &outputName) {
  auto appended = workspaces[0];
  for (auto i = 1u; i < workspaces.size(); ++i)
    appended = appendWorkspace(appended, workspaces[i], outputName);
  return AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>(appended);
}

std::vector<std::string> subdivideWidthWorkspace(const MatrixWorkspace_sptr &workspace,
                                                 const std::vector<std::size_t> &widthSpectra) {
  std::vector<std::string> subworkspaces;
  subworkspaces.reserve(1 + 2 * widthSpectra.size());

  int start = 0;
  for (const auto &spectrum_number : widthSpectra) {
    const auto spectrum = static_cast<int>(spectrum_number);
    if (spectrum > start) {
      auto const outputName = "__extracted_" + std::to_string(start) + "_to_" + std::to_string(spectrum);
      subworkspaces.emplace_back(extractSpectra(workspace->getName(), start, spectrum - 1, outputName));
    }
    subworkspaces.emplace_back(extractHWHMSpectrum(workspace, spectrum));
    start = spectrum + 1;
  }

  const int end = static_cast<int>(workspace->getNumberHistograms());
  if (start < end) {
    auto const outputName = "__extracted_" + std::to_string(start) + "_to_" + std::to_string(end);
    subworkspaces.emplace_back(extractSpectra(workspace->getName(), start, end - 1, outputName));
  }
  return subworkspaces;
}

MatrixWorkspace_sptr createHWHMWorkspace(MatrixWorkspace_sptr workspace, const std::string &hwhmName,
                                         const std::vector<std::size_t> &widthSpectra) {
  if (widthSpectra.empty())
    return workspace;
  if (AnalysisDataService::Instance().doesExist(hwhmName))
    return AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>(hwhmName);

  const auto subworkspaces = subdivideWidthWorkspace(workspace, widthSpectra);
  const auto hwhmWorkspace = appendAll(subworkspaces, hwhmName);
  auto axis = dynamic_cast<TextAxis *>(workspace->getAxis(1)->clone(hwhmWorkspace.get()));
  hwhmWorkspace->replaceAxis(1, std::unique_ptr<TextAxis>(axis));

  deleteTemporaryWorkspaces(subworkspaces);

  return hwhmWorkspace;
}

bool hasParameterSpectra(const FunctionQParameters &parameters) {
  return !parameters.widthSpectra.empty() || !parameters.eisfSpectra.empty();
}

} // namespace

namespace MantidQt::CustomInterfaces::Inelastic {

FunctionQDataPresenter::FunctionQDataPresenter(IFitTab *tab, IDataModel *model, IFitDataView *view)
    : FitDataPresenter(tab, model, view), m_activeParameterType("Width"), m_activeWorkspaceID(WorkspaceID{0}),
      m_adsInstance(Mantid::API::AnalysisDataService::Instance()) {}

bool FunctionQDataPresenter::addWorkspaceFromDialog(MantidWidgets::IAddWorkspaceDialog const *dialog) {
  if (const auto functionQDialog = dynamic_cast<FunctionQAddWorkspaceDialog const *>(dialog)) {
    addWorkspace(functionQDialog->workspaceName(), functionQDialog->parameterType(),
                 functionQDialog->parameterNameIndex());
    setActiveWorkspaceIDToCurrentWorkspace(functionQDialog);

    auto const parameterIndex = functionQDialog->parameterNameIndex();
    if (parameterIndex < 0) {
      throw std::runtime_error("No valid parameter was selected.");
    }
    const auto parameters = createFunctionQParameters(m_model->getWorkspace(m_activeWorkspaceID));
    setActiveSpectra(parameters.spectra(functionQDialog->parameterType()), static_cast<std::size_t>(parameterIndex),
                     m_activeWorkspaceID, false);
    updateActiveWorkspaceID(getNumberOfWorkspaces());
    return true;
  }
  return false;
}

void FunctionQDataPresenter::addWorkspace(const std::string &workspaceName, const std::string &paramType,
                                          const int &spectrum_index) {
  const auto workspace = Mantid::API::AnalysisDataService::Instance().retrieveWS<MatrixWorkspace>(workspaceName);
  const auto parameters = createFunctionQParameters(workspace);
  const auto functionQFunctionList = chooseFunctionQFunctions(paramType == "Width");

  if (!hasParameterSpectra(parameters))
    throw std::invalid_argument("Workspace contains no Width or EISF spectra.");

  if (workspace->y(0).size() == 1)
    throw std::invalid_argument("Workspace contains only one data point.");

  const auto name = getHWHMName(workspace->getName());
  const auto hwhmWorkspace = createHWHMWorkspace(workspace, name, parameters.widthSpectra);
  m_tab->handleFunctionListChanged(functionQFunctionList);
  const auto singleSpectra =
      FunctionModelSpectra(std::to_string(parameters.spectra(m_activeParameterType)[spectrum_index]));
  m_model->addWorkspace(hwhmWorkspace->getName(), singleSpectra);
}

std::map<std::string, std::string> FunctionQDataPresenter::chooseFunctionQFunctions(bool paramWidth) const {
  if (m_view->isTableEmpty()) // when first data is added to table, it can only be either WIDTH or EISF
    return paramWidth ? FunctionQ::WIDTH_FITS : FunctionQ::EISF_FITS;

  bool widthFuncs = paramWidth || m_view->dataColumnContainsText("FWHM");
  bool eisfFuncs = !paramWidth || m_view->dataColumnContainsText("EISF");
  if (widthFuncs && eisfFuncs)
    return FunctionQ::ALL_FITS;
  else if (widthFuncs)
    return FunctionQ::WIDTH_FITS;
  else
    return FunctionQ::EISF_FITS;
}

void FunctionQDataPresenter::setActiveParameterType(const std::string &type) { m_activeParameterType = type; }

void FunctionQDataPresenter::updateActiveWorkspaceID(WorkspaceID index) { m_activeWorkspaceID = index; }

void FunctionQDataPresenter::handleAddClicked() { updateActiveWorkspaceID(m_model->getNumberOfWorkspaces()); }

void FunctionQDataPresenter::handleWorkspaceChanged(FunctionQAddWorkspaceDialog *dialog,
                                                    const std::string &workspaceName) {
  FunctionQParameters parameters;
  try {
    auto workspace = m_adsInstance.retrieveWS<MatrixWorkspace>(workspaceName);
    parameters = createFunctionQParameters(workspace);
    dialog->enableParameterSelection();
  } catch (const std::invalid_argument &) {
    dialog->disableParameterSelection();
  }
  updateParameterTypes(dialog, parameters);
  updateParameterOptions(dialog, parameters);
}

void FunctionQDataPresenter::handleParameterTypeChanged(FunctionQAddWorkspaceDialog *dialog, const std::string &type) {
  const auto workspaceName = dialog->workspaceName();
  if (!workspaceName.empty() && m_adsInstance.doesExist(workspaceName)) {
    auto workspace = m_adsInstance.retrieveWS<MatrixWorkspace>(workspaceName);
    const FunctionQParameters parameter = createFunctionQParameters(workspace);
    setActiveParameterType(type);
    updateParameterOptions(dialog, parameter);
  }
}

void FunctionQDataPresenter::updateParameterOptions(FunctionQAddWorkspaceDialog *dialog,
                                                    const FunctionQParameters &parameter) {
  setActiveWorkspaceIDToCurrentWorkspace(dialog);
  setActiveParameterType(dialog->parameterType());
  dialog->setParameterNames(parameter.names(m_activeParameterType));
}

void FunctionQDataPresenter::updateParameterTypes(FunctionQAddWorkspaceDialog *dialog,
                                                  FunctionQParameters const &parameters) {
  setActiveWorkspaceIDToCurrentWorkspace(dialog);
  dialog->setParameterTypes(getParameterTypes(parameters));
}

std::vector<std::string> FunctionQDataPresenter::getParameterTypes(FunctionQParameters const &parameters) const {
  std::vector<std::string> types;
  if (!parameters.widths.empty())
    types.emplace_back("Width");
  if (!parameters.eisf.empty())
    types.emplace_back("EISF");
  return types;
}

void FunctionQDataPresenter::setActiveWorkspaceIDToCurrentWorkspace(MantidWidgets::IAddWorkspaceDialog const *dialog) {
  //  update active data index with correct index based on the workspace name
  //  and the vector in m_fitDataModel which is in the base class
  //  FittingModel get table workspace index
  const auto wsName = dialog->workspaceName().append("_HWHM");
  // This a vector of workspace names currently loaded
  auto wsVector = m_model->getWorkspaceNames();
  // this is an iterator pointing to the current wsName in wsVector
  auto wsIt = std::find(wsVector.begin(), wsVector.end(), wsName);
  // this is the index of the workspace.
  const auto index = WorkspaceID(std::distance(wsVector.begin(), wsIt));
  updateActiveWorkspaceID(index);
}

void FunctionQDataPresenter::setActiveSpectra(std::vector<std::size_t> const &activeParameterSpectra,
                                              std::size_t parameterIndex, WorkspaceID workspaceID, bool single) {
  if (activeParameterSpectra.size() <= parameterIndex) {
    return;
  }
  if (single) {
    m_model->setSpectra(createSpectra(std::vector<std::size_t>({activeParameterSpectra[parameterIndex]})), workspaceID);
    return;
  }
  // In multiple mode the spectra needs to be appending on the existing spectra list.
  auto spectra = std::vector<std::size_t>({activeParameterSpectra[parameterIndex]});
  for (const auto &i : m_model->getSpectra(workspaceID)) {
    if ((std::find(spectra.begin(), spectra.end(), i.value) == spectra.end())) {
      spectra.emplace_back(i.value);
    }
  }
  m_model->setSpectra(createSpectra(spectra), workspaceID);
}

void FunctionQDataPresenter::addTableEntry(FitDomainIndex row) {
  const auto &name = m_model->getWorkspace(row)->getName();

  const auto subIndices = m_model->getSubIndices(row);
  const auto workspace = m_model->getWorkspace(subIndices.first);
  const auto axis = dynamic_cast<Mantid::API::TextAxis const *>(workspace->getAxis(1));
  if (!axis) {
    return;
  }
  const auto parameter = axis->label(subIndices.second.value);

  const auto workspaceIndex = m_model->getSpectrum(row);
  const auto range = m_model->getFittingRange(row);
  const auto exclude = m_model->getExcludeRegion(row);

  FitDataRow newRow;
  newRow.name = name;
  newRow.workspaceIndex = workspaceIndex;
  newRow.parameter = parameter;
  newRow.startX = range.first;
  newRow.endX = range.second;
  newRow.exclude = exclude;
  m_view->addTableEntry(row.value, newRow);
}
} // namespace MantidQt::CustomInterfaces::Inelastic
