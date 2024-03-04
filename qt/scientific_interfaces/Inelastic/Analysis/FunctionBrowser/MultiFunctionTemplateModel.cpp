// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#include "MultiFunctionTemplateModel.h"
#include "MantidAPI/FunctionFactory.h"
#include "MantidAPI/IFunction.h"
#include "MantidAPI/ITableWorkspace.h"
#include "MantidAPI/MultiDomainFunction.h"
#include "MantidQtWidgets/Common/FunctionBrowser/FunctionBrowserUtils.h"

#include <map>

namespace {
using namespace MantidQt::CustomInterfaces::IDA;

double constexpr DEFAULT_TEMPERATURE_CORRECTION = 100.0;

auto const lorentzian = [](Mantid::MantidVec const &x, Mantid::MantidVec const &y) {
  return std::unordered_map<std::string, double>{{"Amplitude", y[1]}, {"FWHM", 2.0 * std::abs(x[1] - x[0])}};
};

auto const sqeFunction = [](Mantid::MantidVec const &x, Mantid::MantidVec const &y) {
  (void)x;
  return std::unordered_map<std::string, double>{{"Height", y[1]}};
};

auto const estimators = std::unordered_map<std::string, IDAFunctionParameterEstimation::ParameterEstimator>{
    {"Lorentzian", lorentzian},        {"LorentzianN", lorentzian},       {"TeixeiraWaterSQE", sqeFunction},
    {"FickDiffusionSQE", sqeFunction}, {"ChudleyElliotSQE", sqeFunction}, {"HallRossSQE", sqeFunction}};

} // namespace

namespace MantidQt::CustomInterfaces::IDA {

using namespace MantidWidgets;
using namespace Mantid::API;

MultiFunctionTemplateModel::MultiFunctionTemplateModel()
    : m_parameterEstimation(std::make_unique<IDAFunctionParameterEstimation>(estimators)) {}

void MultiFunctionTemplateModel::clearData() {
  m_lorentzianType = LorentzianType::None;
  m_fitType = FitType::None;
  m_deltaType = DeltaType::None;
  m_tempCorrectionType = TempCorrectionType::None;
  m_backgroundType = BackgroundType::None;
  m_model.clear();
}

void MultiFunctionTemplateModel::setModel() {
  m_model.setModel(buildBackgroundFunctionString(), m_fitResolutions, buildLorentzianPeaksString(),
                   buildFitTypeString(), hasDeltaFunction(), m_qValues, m_isQDependentFunction, hasTempCorrection(),
                   DEFAULT_TEMPERATURE_CORRECTION);
  if (hasTempCorrection() && !m_globals.contains(ParamID::TEMPERATURE)) {
    m_globals.push_back(ParamID::TEMPERATURE);
  }
  m_model.setGlobalParameters(makeGlobalList());
  estimateFunctionParameters();
}

void MultiFunctionTemplateModel::setFunction(IFunction_sptr fun) {
  clearData();
  if (!fun)
    return;
  bool isBackgroundSet = false;
  if (fun->name() == "Convolution") {
    checkConvolution(fun);
  } else if (fun->name() == "CompositeFunction") {
    for (size_t i = 0; i < fun->nFunctions(); ++i) {
      auto innerFunction = fun->getFunction(i);
      auto const name = innerFunction->name();
      if (name == "FlatBackground") {
        if (isBackgroundSet) {
          throw std::runtime_error("Function has wrong structure.");
        }
        m_backgroundType = BackgroundType::Flat;
        isBackgroundSet = true;
      } else if (name == "LinearBackground") {
        if (isBackgroundSet) {
          throw std::runtime_error("Function has wrong structure.");
        }
        m_backgroundType = BackgroundType::Linear;
        isBackgroundSet = true;
      } else if (name == "Convolution") {
        checkConvolution(innerFunction);
      }
    }
  }
  m_model.setFunction(fun);
}

void MultiFunctionTemplateModel::checkConvolution(const IFunction_sptr &fun) {
  bool isFitTypeSet = false;
  bool isResolutionSet = false;
  bool isLorentzianTypeSet = false;
  for (size_t i = 0; i < fun->nFunctions(); ++i) {
    auto innerFunction = fun->getFunction(i);
    auto const name = innerFunction->name();
    if (name == "Resolution") {
      if (isResolutionSet) {
        throw std::runtime_error("Function has wrong structure.");
      }
      isResolutionSet = true;
    } else if (name == "ProductFunction") {
      if (innerFunction->getFunction(0)->name() != "ConvTempCorrection" ||
          innerFunction->getFunction(0)->nParams() != 1 ||
          !innerFunction->getFunction(0)->hasParameter("Temperature")) {
        throw std::runtime_error("Function has wrong structure.");
      }
      if (std::dynamic_pointer_cast<CompositeFunction>(innerFunction->getFunction(1)))
        checkConvolution(innerFunction->getFunction(1));
      else
        checkSingleFunction(innerFunction->getFunction(1), isLorentzianTypeSet, isFitTypeSet);
    }

    else if (name == "CompositeFunction") {
      checkConvolution(innerFunction);
    } else {
      checkSingleFunction(innerFunction, isLorentzianTypeSet, isFitTypeSet);
    }
  }
}

void MultiFunctionTemplateModel::checkSingleFunction(const IFunction_sptr &fun, bool &isLorentzianTypeSet,
                                                     bool &isFitTypeSet) {
  auto const name = fun->name();
  if (name == "Lorentzian") {
    if (isLorentzianTypeSet && m_lorentzianType != LorentzianType::OneLorentzian) {
      throw std::runtime_error("Function has wrong structure.");
    }
    if (isLorentzianTypeSet)
      m_lorentzianType = LorentzianType::TwoLorentzians;
    else
      m_lorentzianType = LorentzianType::OneLorentzian;
    isLorentzianTypeSet = true;
  }

  if (FitTypeStringToEnum.count(name) == 1) {
    if (isFitTypeSet) {
      throw std::runtime_error("Function has wrong structure. More than one fit type set");
    }
    m_fitType = FitTypeStringToEnum[name];
    m_isQDependentFunction = FitTypeQDepends[m_fitType];
    isFitTypeSet = true;
  } else if (!isFitTypeSet && !isLorentzianTypeSet) {
    clear();
    throw std::runtime_error("Function has wrong structure. Function not recognized");
  }
}

IFunction_sptr MultiFunctionTemplateModel::getFullFunction() const { return m_model.getFullFunction(); }

IFunction_sptr MultiFunctionTemplateModel::getFitFunction() const { return m_model.getFitFunction(); }

void MultiFunctionTemplateModel::setQValues(const std::vector<double> &qValues) { m_qValues = qValues; }

bool MultiFunctionTemplateModel::hasFunction() const { return m_model.hasFunction(); }

void MultiFunctionTemplateModel::addFunction(std::string const &prefix, std::string const &funStr) {
  if (!prefix.empty())
    throw std::runtime_error("Function doesn't have member function with prefix " + prefix);
  auto fun = FunctionFactory::Instance().createInitialized(funStr);
  auto const name = fun->name();
  std::string newPrefix;
  if (name == "Lorentzian") {
    if (m_lorentzianType == LorentzianType::TwoLorentzians) {
      throw std::runtime_error("Cannot add more Lorentzians.");
    } else if (m_lorentzianType == LorentzianType::OneLorentzian) {
      m_lorentzianType = LorentzianType::TwoLorentzians;
      newPrefix = *getLor2Prefix();
    } else if (m_lorentzianType == LorentzianType::None) {
      m_lorentzianType = LorentzianType::OneLorentzian;
      newPrefix = *getLor1Prefix();
    } else {
      throw std::runtime_error("Cannot add more Lorentzians.");
    }
  } else if (name == "DeltaFunction") {
    if (hasDeltaFunction())
      throw std::runtime_error("Cannot add a DeltaFunction.");
    setSubType(ConvTypes::SubTypeIndex::Delta, static_cast<int>(DeltaType::Delta));
    newPrefix = *getDeltaPrefix();
  } else if (name == "FlatBackground" || name == "LinearBackground") {
    if (hasBackground())
      throw std::runtime_error("Cannot add more backgrounds.");
    if (name == "FlatBackground") {
      setSubType(ConvTypes::SubTypeIndex::Background, static_cast<int>(BackgroundType::Flat));
    } else if (name == "LinearBackground") {
      setSubType(ConvTypes::SubTypeIndex::Background, static_cast<int>(BackgroundType::Linear));
    }
    newPrefix = *getBackgroundPrefix();
  } else {
    throw std::runtime_error("Cannot add funtion " + name);
  }
  auto newFun = getFunctionWithPrefix(newPrefix, getSingleFunction(0));
  copyParametersAndErrors(*fun, *newFun);
  if (getNumberLocalFunctions() > 1) {
    copyParametersAndErrorsToAllLocalFunctions(*getSingleFunction(0));
  }
}

void MultiFunctionTemplateModel::removeFunction(std::string const &prefix) {
  if (prefix.empty()) {
    clear();
    return;
  }
  auto prefix1 = getLor1Prefix();
  if (prefix1 && *prefix1 == prefix) {
    setSubType(ConvTypes::SubTypeIndex::Lorentzian, static_cast<int>(LorentzianType::None));
    return;
  }
  prefix1 = getLor2Prefix();
  if (prefix1 && *prefix1 == prefix) {
    setSubType(ConvTypes::SubTypeIndex::Lorentzian, static_cast<int>(LorentzianType::OneLorentzian));
    return;
  }
  prefix1 = getDeltaPrefix();
  if (prefix1 && *prefix1 == prefix) {
    setSubType(ConvTypes::SubTypeIndex::Delta, static_cast<int>(DeltaType::None));
    return;
  }
  prefix1 = getBackgroundPrefix();
  if (prefix1 && *prefix1 == prefix) {
    removeBackground();
    return;
  }
  throw std::runtime_error("Function doesn't have member function with prefix " + prefix);
}

bool MultiFunctionTemplateModel::hasTempCorrection() const { return m_tempCorrectionType != TempCorrectionType::None; }

bool MultiFunctionTemplateModel::hasDeltaFunction() const { return m_deltaType != DeltaType::None; }

void MultiFunctionTemplateModel::removeBackground() {
  auto oldValues = getCurrentValues();
  m_backgroundType = BackgroundType::None;
  setModel();
  setCurrentValues(oldValues);
}

bool MultiFunctionTemplateModel::hasBackground() const { return m_backgroundType != BackgroundType::None; }

EstimationDataSelector MultiFunctionTemplateModel::getEstimationDataSelector() const {
  return [](const Mantid::MantidVec &x, const Mantid::MantidVec &y,
            const std::pair<double, double> &range) -> DataForParameterEstimation {
    (void)range;

    auto const maxElement = std::max_element(y.cbegin(), y.cend());
    auto const halfMaxElement =
        std::find_if(y.cbegin(), y.cend(), [&maxElement](double const val) { return val > *maxElement / 2.0; });
    if (maxElement == y.cend() || halfMaxElement == y.cend())
      return DataForParameterEstimation{{}, {}};

    auto const maxElementIndex = std::distance(y.cbegin(), maxElement);
    auto const halfMaxElementIndex = std::distance(y.cbegin(), halfMaxElement);

    return DataForParameterEstimation{{x[halfMaxElementIndex], x[maxElementIndex]}, {*halfMaxElement, *maxElement}};
  };
}

void MultiFunctionTemplateModel::updateParameterEstimationData(DataForParameterEstimationCollection &&data) {
  m_estimationData = std::move(data);
}

void MultiFunctionTemplateModel::estimateFunctionParameters() {
  m_parameterEstimation->estimateFunctionParameters(getFullFunction(), m_estimationData);
}

void MultiFunctionTemplateModel::setResolution(const std::vector<std::pair<std::string, size_t>> &fitResolutions) {
  m_fitResolutions = fitResolutions;
  setModel();
}

std::string MultiFunctionTemplateModel::setBackgroundA0(double value) {
  if (hasBackground()) {
    auto const paramID = (m_backgroundType == BackgroundType::Flat) ? ParamID::FLAT_BG_A0 : ParamID::LINEAR_BG_A0;
    setParameter(paramID, value);
    return *getParameterName(paramID);
  }
  return "";
}

void MultiFunctionTemplateModel::setNumberDomains(int n) { m_model.setNumberDomains(n); }

int MultiFunctionTemplateModel::getNumberDomains() const { return m_model.getNumberDomains(); }

void MultiFunctionTemplateModel::setParameter(std::string const &parameterName, double value) {
  m_model.setParameter(parameterName, value);
}

void MultiFunctionTemplateModel::setParameterError(std::string const &parameterName, double value) {
  m_model.setParameterError(parameterName, value);
}

double MultiFunctionTemplateModel::getParameter(std::string const &parameterName) const {
  return m_model.getParameter(parameterName);
}

double MultiFunctionTemplateModel::getParameterError(std::string const &parameterName) const {
  return m_model.getParameterError(parameterName);
}

std::string MultiFunctionTemplateModel::getParameterDescription(std::string const &parameterName) const {
  return m_model.getParameterDescription(parameterName);
}

std::vector<std::string> MultiFunctionTemplateModel::getParameterNames() const { return m_model.getParameterNames(); }

IFunction_sptr MultiFunctionTemplateModel::getSingleFunction(int index) const {
  return m_model.getSingleFunction(index);
}

IFunction_sptr MultiFunctionTemplateModel::getCurrentFunction() const { return m_model.getCurrentFunction(); }

std::vector<std::string> MultiFunctionTemplateModel::getGlobalParameters() const {
  return m_model.getGlobalParameters();
}

std::vector<std::string> MultiFunctionTemplateModel::getLocalParameters() const { return m_model.getLocalParameters(); }

void MultiFunctionTemplateModel::setGlobalParameters(std::vector<std::string> const &globals) {
  m_globals.clear();
  for (auto const &name : globals) {
    addGlobal(name);
  }
  auto newGlobals = makeGlobalList();
  m_model.setGlobalParameters(newGlobals);
}

bool MultiFunctionTemplateModel::isGlobal(std::string const &parameterName) const {
  return m_model.isGlobal(parameterName);
}

void MultiFunctionTemplateModel::setGlobal(std::string const &parameterName, bool on) {
  if (parameterName.empty())
    return;
  if (on)
    addGlobal(parameterName);
  else
    removeGlobal(parameterName);
  auto globals = makeGlobalList();
  m_model.setGlobalParameters(globals);
}

void MultiFunctionTemplateModel::addGlobal(std::string const &parameterName) {
  auto const pid = getParameterId(parameterName);
  if (pid && !m_globals.contains(*pid)) {
    m_globals.push_back(*pid);
  }
}

void MultiFunctionTemplateModel::removeGlobal(std::string const &parameterName) {
  auto const pid = getParameterId(parameterName);
  if (pid && m_globals.contains(*pid)) {
    m_globals.removeOne(*pid);
  }
}

std::vector<std::string> MultiFunctionTemplateModel::makeGlobalList() const {
  std::vector<std::string> globals;
  for (auto const id : m_globals) {
    auto const name = getParameterName(id);
    if (name)
      globals.emplace_back(*name);
  }
  return globals;
}

void MultiFunctionTemplateModel::setSubType(std::size_t subTypeIndex, int typeIndex) {
  auto oldValues = getCurrentValues();
  if (subTypeIndex == ConvTypes::SubTypeIndex::Fit) {
    m_fitType = static_cast<ConvTypes::FitType>(typeIndex);
    m_isQDependentFunction = FitTypeQDepends[m_fitType];
  } else if (subTypeIndex == ConvTypes::SubTypeIndex::Lorentzian) {
    m_lorentzianType = static_cast<ConvTypes::LorentzianType>(typeIndex);
  } else if (subTypeIndex == ConvTypes::SubTypeIndex::Delta) {
    m_deltaType = static_cast<ConvTypes::DeltaType>(typeIndex);
  } else if (subTypeIndex == ConvTypes::SubTypeIndex::TempCorrection) {
    m_tempCorrectionType = static_cast<ConvTypes::TempCorrectionType>(typeIndex);
  } else {
    m_backgroundType = static_cast<ConvTypes::BackgroundType>(typeIndex);
  }
  setModel();
  setCurrentValues(oldValues);
}

std::map<std::size_t, int> MultiFunctionTemplateModel::getSubTypes() const {
  std::map<std::size_t, int> subTypes;
  subTypes[ConvTypes::SubTypeIndex::Lorentzian] = static_cast<int>(m_lorentzianType);
  subTypes[ConvTypes::SubTypeIndex::Fit] = static_cast<int>(m_fitType);
  subTypes[ConvTypes::SubTypeIndex::Delta] = static_cast<int>(m_deltaType);
  subTypes[ConvTypes::SubTypeIndex::TempCorrection] = static_cast<int>(m_tempCorrectionType);
  subTypes[ConvTypes::SubTypeIndex::Background] = static_cast<int>(m_backgroundType);
  return subTypes;
}

int MultiFunctionTemplateModel::getNumberOfPeaks() const {
  if (m_lorentzianType == LorentzianType::None)
    return 0;
  if (m_lorentzianType == LorentzianType::TwoLorentzians)
    return 2;
  return 1;
}

void MultiFunctionTemplateModel::updateMultiDatasetParameters(const IFunction &fun) {
  m_model.updateMultiDatasetParameters(fun);
}

void MultiFunctionTemplateModel::updateMultiDatasetParameters(const ITableWorkspace &paramTable) {
  auto const nRows = paramTable.rowCount();
  if (nRows == 0)
    return;

  auto const globalParameterNames = getGlobalParameters();
  for (auto &&name : globalParameterNames) {
    auto valueColumn = paramTable.getColumn(name);
    auto errorColumn = paramTable.getColumn(name + "_Err");
    m_model.setParameter(name, valueColumn->toDouble(0));
    m_model.setParameterError(name, errorColumn->toDouble(0));
  }

  auto const localParameterNames = getLocalParameters();
  for (auto &&name : localParameterNames) {
    auto valueColumn = paramTable.getColumn(name);
    auto errorColumn = paramTable.getColumn(name + "_Err");
    if (nRows > 1) {
      for (size_t i = 0; i < nRows; ++i) {
        m_model.setLocalParameterValue(name, static_cast<int>(i), valueColumn->toDouble(i), errorColumn->toDouble(i));
      }
    } else {
      auto const i = m_model.currentDomainIndex();
      m_model.setLocalParameterValue(name, static_cast<int>(i), valueColumn->toDouble(0), errorColumn->toDouble(0));
    }
  }
}

void MultiFunctionTemplateModel::updateParameters(const IFunction &fun) { m_model.updateParameters(fun); }

void MultiFunctionTemplateModel::setCurrentDomainIndex(int i) { m_model.setCurrentDomainIndex(i); }

int MultiFunctionTemplateModel::currentDomainIndex() const { return m_model.currentDomainIndex(); }

void MultiFunctionTemplateModel::changeTie(std::string const &parameterName, std::string const &tie) {
  m_model.changeTie(parameterName, tie);
}

void MultiFunctionTemplateModel::addConstraint(std::string const &functionIndex, std::string const &constraint) {
  m_model.addConstraint(functionIndex, constraint);
}

void MultiFunctionTemplateModel::removeConstraint(std::string const &parameterName) {
  m_model.removeConstraint(parameterName);
}

void MultiFunctionTemplateModel::setDatasets(const QList<FunctionModelDataset> &datasets) {
  m_model.setDatasets(datasets);
}

QStringList MultiFunctionTemplateModel::getDatasetNames() const { return m_model.getDatasetNames(); }

QStringList MultiFunctionTemplateModel::getDatasetDomainNames() const { return m_model.getDatasetDomainNames(); }

double MultiFunctionTemplateModel::getLocalParameterValue(std::string const &parameterName, int i) const {
  return m_model.getLocalParameterValue(parameterName, i);
}

bool MultiFunctionTemplateModel::isLocalParameterFixed(std::string const &parameterName, int i) const {
  return m_model.isLocalParameterFixed(parameterName, i);
}

std::string MultiFunctionTemplateModel::getLocalParameterTie(std::string const &parameterName, int i) const {
  return m_model.getLocalParameterTie(parameterName, i);
}

std::string MultiFunctionTemplateModel::getLocalParameterConstraint(std::string const &parameterName, int i) const {
  return m_model.getLocalParameterConstraint(parameterName, i);
}

void MultiFunctionTemplateModel::setLocalParameterValue(std::string const &parameterName, int i, double value) {
  m_model.setLocalParameterValue(parameterName, i, value);
}

void MultiFunctionTemplateModel::setLocalParameterValue(std::string const &parameterName, int i, double value,
                                                        double error) {
  m_model.setLocalParameterValue(parameterName, i, value, error);
}

void MultiFunctionTemplateModel::setLocalParameterTie(std::string const &parameterName, int i, std::string const &tie) {
  m_model.setLocalParameterTie(parameterName, i, tie);
}

void MultiFunctionTemplateModel::setLocalParameterConstraint(std::string const &parameterName, int i,
                                                             std::string const &constraint) {
  m_model.setLocalParameterConstraint(parameterName, i, constraint);
}

void MultiFunctionTemplateModel::setLocalParameterFixed(std::string const &parameterName, int i, bool fixed) {
  m_model.setLocalParameterFixed(parameterName, i, fixed);
}

void MultiFunctionTemplateModel::setGlobalParameterValue(std::string const &parameterName, double value) {
  m_model.setGlobalParameterValue(parameterName, value);
}

void MultiFunctionTemplateModel::setParameter(ParamID name, double value) {
  auto const prefix = getPrefix(name);
  if (prefix) {
    m_model.setParameter(*prefix + g_paramName.at(name), value);
  }
}

boost::optional<double> MultiFunctionTemplateModel::getParameter(ParamID name) const {
  auto const paramName = getParameterName(name);
  return paramName ? m_model.getParameter(*paramName) : boost::optional<double>();
}

boost::optional<double> MultiFunctionTemplateModel::getParameterError(ParamID name) const {
  auto const paramName = getParameterName(name);
  return paramName ? m_model.getParameterError(*paramName) : boost::optional<double>();
}

boost::optional<std::string> MultiFunctionTemplateModel::getParameterName(ParamID name) const {
  auto const prefix = getPrefix(name);
  return prefix ? *prefix + g_paramName.at(name) : boost::optional<std::string>();
}

boost::optional<std::string> MultiFunctionTemplateModel::getParameterDescription(ParamID name) const {
  auto const paramName = getParameterName(name);
  return paramName ? m_model.getParameterDescription(*paramName) : boost::optional<std::string>();
}

boost::optional<std::string> MultiFunctionTemplateModel::getPrefix(ParamID name) const {
  if (name >= ParamID::FLAT_BG_A0) {
    return m_model.backgroundPrefix();
  } else if (name == ParamID::DELTA_HEIGHT || name == ParamID::DELTA_CENTER) {
    return m_model.deltaFunctionPrefix();
  } else if (name == ParamID::TEMPERATURE) {
    return m_model.tempFunctionPrefix();
  } else if (name >= ParamID::TW_HEIGHT && name < ParamID::FLAT_BG_A0) {
    return m_model.fitTypePrefix();
  } else {
    auto const prefixes = m_model.peakPrefixes();
    if (!prefixes)
      return boost::optional<std::string>();
    auto const index = name > ParamID::LOR2_FWHM_1 && name <= ParamID::LOR2_FWHM_2 ? 1 : 0;
    return m_model.peakPrefixes()->at(index).toStdString();
  }
}

QMap<ParamID, double> MultiFunctionTemplateModel::getCurrentValues() const {
  QMap<ParamID, double> values;
  auto store = [&values, this](ParamID name) { values[name] = *getParameter(name); };
  applyParameterFunction(store);
  return values;
}

QMap<ParamID, double> MultiFunctionTemplateModel::getCurrentErrors() const {
  QMap<ParamID, double> errors;
  auto store = [&errors, this](ParamID name) { errors[name] = *getParameterError(name); };
  applyParameterFunction(store);
  return errors;
}

QMap<int, std::string> MultiFunctionTemplateModel::getParameterNameMap() const {
  QMap<int, std::string> out;
  auto addToMap = [&out, this](ParamID name) { out[static_cast<int>(name)] = *getParameterName(name); };
  applyParameterFunction(addToMap);
  return out;
}

void MultiFunctionTemplateModel::setCurrentValues(const QMap<ParamID, double> &values) {
  for (auto const name : values.keys()) {
    setParameter(name, values[name]);
  }
}

void MultiFunctionTemplateModel::applyParameterFunction(const std::function<void(ParamID)> &paramFun) const {
  applyToFitFunction<ConvTypes::LorentzianSubType>(m_lorentzianType, paramFun);
  applyToFitFunction<ConvTypes::FitSubType>(m_fitType, paramFun);
  applyToFitFunction<ConvTypes::DeltaSubType>(hasDeltaFunction(), paramFun);
  applyToFitFunction<ConvTypes::TempSubType>(m_tempCorrectionType, paramFun);
  applyToFitFunction<ConvTypes::BackgroundSubType>(m_backgroundType, paramFun);
}

boost::optional<ParamID> MultiFunctionTemplateModel::getParameterId(std::string const &parameterName) {
  boost::optional<ParamID> result;
  auto getter = [&result, parameterName, this](ParamID pid) {
    if (parameterName == *getParameterName(pid))
      result = pid;
  };
  applyParameterFunction(getter);
  return result;
}

std::string MultiFunctionTemplateModel::buildLorentzianFunctionString() const {
  return "name=Lorentzian,Amplitude=1,FWHM=1,constraints=(Amplitude>0,FWHM>"
         "0)";
}

std::string MultiFunctionTemplateModel::buildTeixeiraFunctionString() const {
  return "name=TeixeiraWaterSQE, Height=1, DiffCoeff=2.3, Tau=1.25, Centre=0, "
         "constraints=(Height>0, DiffCoeff>0, Tau>0)";
}

std::string MultiFunctionTemplateModel::buildFickFunctionString() const {
  return "name=FickDiffusionSQE, Height=1, DiffCoeff=2.3, Centre=0, "
         "constraints=(Height>0, DiffCoeff>0)";
}

std::string MultiFunctionTemplateModel::buildChudleyElliotString() const {
  return "name=ChudleyElliotSQE, Height=1, Tau=1.25, Centre=0, L=1.0, "
         "constraints=(Height>0, Tau>0, L>0)";
}

std::string MultiFunctionTemplateModel::buildHallRossString() const {
  return "name=HallRossSQE, Height=1, Tau=1.25, Centre=0, L=1.0, "
         "constraints=(Height>0, Tau>0, L>0)";
}

std::string MultiFunctionTemplateModel::buildStretchExpFTFunctionString() const {
  return "name=StretchedExpFT, Height=0.1, Tau=100, Beta=1, Centre=0, "
         "constraints=(Height>0, Tau>0)";
}

std::string MultiFunctionTemplateModel::buildDiffRotDiscreteCircleFunctionString() const {
  return "name=DiffRotDiscreteCircle, f1.Intensity=1, f1.Radius=1, f1.Decay=1, "
         "f1.Shift=0, constraints=(f1.Intensity>0, f1.Radius>0)";
}

std::string MultiFunctionTemplateModel::buildInelasticDiffRotDiscreteCircleFunctionString() const {
  return "name=InelasticDiffRotDiscreteCircle, Intensity=1, Radius=1, Decay=1, "
         "Shift=0, constraints=(Intensity>0, Radius>0)";
}

std::string MultiFunctionTemplateModel::buildElasticDiffRotDiscreteCircleFunctionString() const {
  return "name=ElasticDiffRotDiscreteCircle, Height=1, Centre=0, Radius=1, "
         "constraints=(Height>0, Radius>0)";
}

std::string MultiFunctionTemplateModel::buildDiffSphereFunctionString() const {
  return "name=DiffSphere, Q=1, f0.Q=1, "
         "f0.WorkspaceIndex=2147483647, f1.Q = 1, f1.WorkspaceIndex = "
         "2147483647, f0.Height = 1, f0.Centre = 0, f0.Radius = 2, "
         "f1.Intensity = 1, f1.Radius = 2, f1.Diffusion = 0.05, f1.Shift = 0";
}

std::string MultiFunctionTemplateModel::buildElasticDiffSphereFunctionString() const {
  return "name=ElasticDiffSphere, Height=1, Centre=0, Radius=2, "
         "constraints=(Height>0, Radius>0)";
}

std::string MultiFunctionTemplateModel::buildInelasticDiffSphereFunctionString() const {
  return "name=InelasticDiffSphere, Intensity=1, Radius=2, Diffusion=0.05, "
         "Shift=0, constraints=(Intensity>0, Radius>0, Diffusion>0)";
}

std::string MultiFunctionTemplateModel::buildIsoRotDiffFunctionString() const {
  return "name=IsoRotDiff, f1.Height=0.1, f1.Radius=2, f1.Tau=100, f1.Centre=0";
}

std::string MultiFunctionTemplateModel::buildElasticIsoRotDiffFunctionString() const {
  return "name=ElasticIsoRotDiff, Height=0.1, Radius=2";
}

std::string MultiFunctionTemplateModel::buildInelasticIsoRotDiffFunctionString() const {
  return "name=InelasticIsoRotDiff, Height=0.1, Radius=2, Tau=100, Centre=0";
}

std::string MultiFunctionTemplateModel::buildPeaksFunctionString() const {
  std::string functions;
  if (m_lorentzianType == LorentzianType::OneLorentzian) {
    functions.append(buildLorentzianFunctionString());
  } else if (m_lorentzianType == LorentzianType::TwoLorentzians) {
    auto const lorentzian = buildLorentzianFunctionString();
    functions.append(lorentzian);
    functions.append(";");
    functions.append(lorentzian);
  }
  if (m_fitType == FitType::TeixeiraWater) {
    functions.append(buildTeixeiraFunctionString());
  } else if (m_fitType == FitType::FickDiffusion) {
    functions.append(buildFickFunctionString());
  } else if (m_fitType == FitType::ChudleyElliot) {
    functions.append(buildChudleyElliotString());
  } else if (m_fitType == FitType::HallRoss) {
    functions.append(buildHallRossString());
  } else if (m_fitType == FitType::StretchedExpFT) {
    functions.append(buildStretchExpFTFunctionString());
  } else if (m_fitType == FitType::DiffSphere) {
    functions.append(buildDiffSphereFunctionString());
  } else if (m_fitType == FitType::ElasticDiffSphere) {
    functions.append(buildElasticDiffSphereFunctionString());
  } else if (m_fitType == FitType::InelasticDiffSphere) {
    functions.append(buildInelasticDiffSphereFunctionString());
  } else if (m_fitType == FitType::DiffRotDiscreteCircle) {
    functions.append(buildDiffRotDiscreteCircleFunctionString());
  } else if (m_fitType == FitType::InelasticDiffRotDiscreteCircle) {
    functions.append(buildInelasticDiffRotDiscreteCircleFunctionString());
  } else if (m_fitType == FitType::ElasticDiffRotDiscreteCircle) {
    functions.append(buildElasticDiffRotDiscreteCircleFunctionString());
  } else if (m_fitType == FitType::IsoRotDiff) {
    functions.append(buildIsoRotDiffFunctionString());
  } else if (m_fitType == FitType::ElasticIsoRotDiff) {
    functions.append(buildElasticIsoRotDiffFunctionString());
  } else if (m_fitType == FitType::InelasticIsoRotDiff) {
    functions.append(buildInelasticIsoRotDiffFunctionString());
  }
  return functions;
}

std::string MultiFunctionTemplateModel::buildLorentzianPeaksString() const {
  std::string functions;
  if (m_lorentzianType == LorentzianType::OneLorentzian) {
    functions.append(buildLorentzianFunctionString());
  } else if (m_lorentzianType == LorentzianType::TwoLorentzians) {
    auto const lorentzian = buildLorentzianFunctionString();
    functions.append(lorentzian);
    functions.append(";");
    functions.append(lorentzian);
  }
  return functions;
}

std::string MultiFunctionTemplateModel::buildFitTypeString() const {
  std::string functions;
  if (m_fitType == FitType::TeixeiraWater) {
    functions.append(buildTeixeiraFunctionString());
  } else if (m_fitType == FitType::FickDiffusion) {
    functions.append(buildFickFunctionString());
  } else if (m_fitType == FitType::ChudleyElliot) {
    functions.append(buildChudleyElliotString());
  } else if (m_fitType == FitType::HallRoss) {
    functions.append(buildHallRossString());
  } else if (m_fitType == FitType::StretchedExpFT) {
    functions.append(buildStretchExpFTFunctionString());
  } else if (m_fitType == FitType::DiffSphere) {
    functions.append(buildDiffSphereFunctionString());
  } else if (m_fitType == FitType::ElasticDiffSphere) {
    functions.append(buildElasticDiffSphereFunctionString());
  } else if (m_fitType == FitType::InelasticDiffSphere) {
    functions.append(buildInelasticDiffSphereFunctionString());
  } else if (m_fitType == FitType::DiffRotDiscreteCircle) {
    functions.append(buildDiffRotDiscreteCircleFunctionString());
  } else if (m_fitType == FitType::InelasticDiffRotDiscreteCircle) {
    functions.append(buildInelasticDiffRotDiscreteCircleFunctionString());
  } else if (m_fitType == FitType::ElasticDiffRotDiscreteCircle) {
    functions.append(buildElasticDiffRotDiscreteCircleFunctionString());
  } else if (m_fitType == FitType::IsoRotDiff) {
    functions.append(buildIsoRotDiffFunctionString());
  } else if (m_fitType == FitType::ElasticIsoRotDiff) {
    functions.append(buildElasticIsoRotDiffFunctionString());
  } else if (m_fitType == FitType::InelasticIsoRotDiff) {
    functions.append(buildInelasticIsoRotDiffFunctionString());
  }
  return functions;
}

std::string MultiFunctionTemplateModel::buildBackgroundFunctionString() const {
  if (m_backgroundType == BackgroundType::None)
    return "";
  return "name=" + m_backgroundSubtype.getFunctionName(m_backgroundType) + ",A0=0,constraints=(A0>0)";
}

boost::optional<std::string> MultiFunctionTemplateModel::getLor1Prefix() const {
  return m_model.peakPrefixes()->at(0).toStdString();
}

boost::optional<std::string> MultiFunctionTemplateModel::getLor2Prefix() const {
  return m_model.peakPrefixes()->at(1).toStdString();
}

boost::optional<std::string> MultiFunctionTemplateModel::getFitTypePrefix() const { return m_model.fitTypePrefix(); }

boost::optional<std::string> MultiFunctionTemplateModel::getDeltaPrefix() const {
  return m_model.deltaFunctionPrefix();
}

boost::optional<std::string> MultiFunctionTemplateModel::getBackgroundPrefix() const {
  return m_model.backgroundPrefix();
}

} // namespace MantidQt::CustomInterfaces::IDA
