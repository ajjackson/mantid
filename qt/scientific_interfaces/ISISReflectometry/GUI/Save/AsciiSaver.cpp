// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "AsciiSaver.h"
#include "MantidAPI/AlgorithmManager.h"
#include "MantidAPI/AnalysisDataService.h"
#include "MantidAPI/Workspace.h"
#include "MantidAPI/WorkspaceGroup.h"
#include <Poco/Path.h>

#include <utility>

namespace MantidQt::CustomInterfaces::ISISReflectometry {

namespace {
bool shouldSaveToSingleFile(FileFormatOptions const &fileFormat) {
  return fileFormat.shouldSaveToSingleFile() && fileFormat.isORSOFormat();
}
} // unnamed namespace

AsciiSaver::AsciiSaver(std::unique_ptr<ISaveAlgorithmRunner> saveAlgRunner, IFileHandler *fileHandler)
    : m_saveAlgRunner(std::move(saveAlgRunner)), m_fileHandler(fileHandler) {}

std::string AsciiSaver::extensionForFormat(NamedFormat format) {
  // For the custom format we need to pass just the word "custom" to the "extension" property of the save algorithm
  switch (format) {
  case NamedFormat::Custom:
    return "custom";
  case NamedFormat::ThreeColumn:
    return ".dat";
  case NamedFormat::ANSTO:
    return ".txt";
  case NamedFormat::ILLCosmos:
    return ".mft";
  case NamedFormat::ORSOAscii:
    return ".ort";
  default:
    throw std::runtime_error("Unknown save format.");
  }
}

bool AsciiSaver::isValidSaveDirectory(std::string const &path) const { return m_fileHandler->fileExists(path); }

std::string AsciiSaver::assembleSavePath(std::string const &saveDirectory, std::string const &prefix,
                                         std::string const &name, std::string const &extension) const {
  auto path = Poco::Path(saveDirectory).makeDirectory();
  // The extension is added automatically except where it is "custom"
  if (extension == "custom")
    path.append(prefix + name + std::string(".dat"));
  else
    path.append(prefix + name);

  return path.toString();
}

Mantid::API::Workspace_sptr AsciiSaver::workspace(std::string const &workspaceName) const {
  auto const &ads = Mantid::API::AnalysisDataService::Instance();

  if (!ads.doesExist(workspaceName))
    return nullptr;

  return ads.retrieveWS<Mantid::API::Workspace>(workspaceName);
}

void AsciiSaver::runSaveAsciiAlgorithm(std::string const &savePath, std::string const &extension,
                                       const Mantid::API::Workspace_sptr &workspace,
                                       std::vector<std::string> const &logParameters,
                                       FileFormatOptions const &fileFormat) const {
  m_saveAlgRunner->runSaveAsciiAlgorithm(workspace, savePath, extension, logParameters,
                                         fileFormat.shouldIncludeHeader(), fileFormat.shouldIncludeQResolution(),
                                         fileFormat.separator());
}

void AsciiSaver::runSaveORSOAlgorithm(std::string const &savePath, std::vector<std::string> const &workspaceNames,
                                      FileFormatOptions const &fileFormat) const {
  m_saveAlgRunner->runSaveORSOAlgorithm(workspaceNames, savePath, fileFormat.shouldIncludeQResolution(),
                                        fileFormat.shouldIncludeAdditionalColumns());
}

void AsciiSaver::save(const Mantid::API::Workspace_sptr &workspace, std::string const &saveDirectory,
                      std::vector<std::string> const &logParameters, FileFormatOptions const &fileFormat) const {
  auto const extension = extensionForFormat(fileFormat.format());
  auto const savePath = assembleSavePath(saveDirectory, fileFormat.prefix(), workspace->getName(), extension);

  if (fileFormat.isORSOFormat()) {
    const std::vector<std::string> workspaceNames{workspace->getName()};
    runSaveORSOAlgorithm(savePath, workspaceNames, fileFormat);
  } else {
    runSaveAsciiAlgorithm(savePath, extension, workspace, logParameters, fileFormat);
  }
}

void AsciiSaver::saveToSingleFile(std::vector<std::string> const &workspaceNames, std::string const &saveDirectory,
                                  std::vector<std::string> const &logParameters,
                                  FileFormatOptions const &fileFormat) const {
  auto const extension = extensionForFormat(fileFormat.format());
  auto const savePath = assembleSavePath(saveDirectory, fileFormat.prefix(), workspaceNames.front(), extension);

  if (fileFormat.isORSOFormat()) {
    runSaveORSOAlgorithm(savePath, workspaceNames, fileFormat);
  } else {
    throw std::invalid_argument(
        "Saving multiple workspaces to a single file is not supported for the selected file format.");
  }
}

void AsciiSaver::save(std::string const &saveDirectory, std::vector<std::string> const &workspaceNames,
                      std::vector<std::string> const &logParameters, FileFormatOptions const &fileFormat) const {
  if (!isValidSaveDirectory(saveDirectory)) {
    throw InvalidSavePath(saveDirectory);
  }

  // Setup the appropriate save algorithm
  if (shouldSaveToSingleFile(fileFormat)) {
    saveToSingleFile(workspaceNames, saveDirectory, logParameters, fileFormat);
  } else {
    for (auto const &name : workspaceNames) {
      auto ws = workspace(name);
      if (ws->isGroup()) {
        // Save child workspaces into separate files
        Mantid::API::WorkspaceGroup_sptr group = std::dynamic_pointer_cast<Mantid::API::WorkspaceGroup>(ws);
        for (auto child : group->getAllItems())
          save(child, saveDirectory, logParameters, fileFormat);
        continue;
      }

      save(ws, saveDirectory, logParameters, fileFormat);
    }
  }
}
} // namespace MantidQt::CustomInterfaces::ISISReflectometry
