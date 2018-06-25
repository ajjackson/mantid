#include "MantidKernel/TimeSeriesProperty.h"
#include "MantidPythonInterface/kernel/Converters/DateAndTime.h"
#include "MantidPythonInterface/kernel/GetPointer.h"
#include "MantidPythonInterface/kernel/Policies/VectorToNumpy.h"
#include "MantidKernel/ContainerDtype.h"

#include <boost/python/class.hpp>
#include <boost/python/implicit.hpp>
#include <boost/python/init.hpp>
#include <boost/python/make_function.hpp>
#include <boost/python/register_ptr_to_python.hpp>
#include <boost/python/return_value_policy.hpp>

using Mantid::Kernel::Property;
using Mantid::Kernel::TimeSeriesProperty;
using Mantid::Types::Core::DateAndTime;
using namespace boost::python;
using boost::python::arg;

GET_POINTER_SPECIALIZATION(TimeSeriesProperty<std::string>)
GET_POINTER_SPECIALIZATION(TimeSeriesProperty<int32_t>)
GET_POINTER_SPECIALIZATION(TimeSeriesProperty<int64_t>)
GET_POINTER_SPECIALIZATION(TimeSeriesProperty<bool>)
GET_POINTER_SPECIALIZATION(TimeSeriesProperty<double>)

namespace {

using Mantid::PythonInterface::Policies::VectorToNumpy;

template <typename TYPE>
void addPyTimeValue(TimeSeriesProperty<TYPE> &self,
                    const boost::python::api::object &datetime,
                    const TYPE &value) {
  const auto dateandtime =
      Mantid::PythonInterface::Converters::to_dateandtime(datetime);
  self.addValue(*dateandtime, value);
}

// Call the dtype helper function
template <typename TYPE> std::string dtype(TimeSeriesProperty<TYPE> &self) {
  return dtypeHelper::dtype(self);
}

// Macro to reduce copy-and-paste
#define EXPORT_TIMESERIES_PROP(TYPE, Prefix)                                   \
  register_ptr_to_python<TimeSeriesProperty<TYPE> *>();                        \
                                                                               \
  class_<TimeSeriesProperty<TYPE>, bases<Property>, boost::noncopyable>(       \
      #Prefix "TimeSeriesProperty",                                            \
      init<const std::string &>((arg("self"), arg("value"))))                  \
      .add_property(                                                           \
           "value",                                                            \
           make_function(                                                      \
               &Mantid::Kernel::TimeSeriesProperty<TYPE>::valuesAsVector,      \
               return_value_policy<VectorToNumpy>()))                          \
      .add_property(                                                           \
           "times",                                                            \
           make_function(                                                      \
               &Mantid::Kernel::TimeSeriesProperty<TYPE>::timesAsVector,       \
               return_value_policy<VectorToNumpy>()))                          \
      .def("addValue", (void (TimeSeriesProperty<TYPE>::*)(                    \
                           const DateAndTime &, const TYPE)) &                 \
                           TimeSeriesProperty<TYPE>::addValue,                 \
           (arg("self"), arg("time"), arg("value")))                           \
      .def("addValue", (void (TimeSeriesProperty<TYPE>::*)(                    \
                           const std::string &, const TYPE)) &                 \
                           TimeSeriesProperty<TYPE>::addValue,                 \
           (arg("self"), arg("time"), arg("value")))                           \
      .def("addValue", &addPyTimeValue<TYPE>,                                  \
           (arg("self"), arg("time"), arg("value")))                           \
      .def("clear", &TimeSeriesProperty<TYPE>::clear, arg("self"))             \
      .def("valueAsString", &TimeSeriesProperty<TYPE>::value, arg("self"))     \
      .def("size", &TimeSeriesProperty<TYPE>::size, arg("self"))               \
      .def("firstTime", &TimeSeriesProperty<TYPE>::firstTime, arg("self"),     \
           "returns :class:`mantid.kernel.DateAndTime`")                       \
      .def("firstValue", &TimeSeriesProperty<TYPE>::firstValue, arg("self"))   \
      .def("lastTime", &TimeSeriesProperty<TYPE>::lastTime, arg("self"),       \
           "returns :class:`mantid.kernel.DateAndTime`")                       \
      .def("lastValue", &TimeSeriesProperty<TYPE>::lastValue, arg("self"))     \
      .def("nthValue", &TimeSeriesProperty<TYPE>::nthValue,                    \
           (arg("self"), arg("index")))                                        \
      .def("nthTime", &TimeSeriesProperty<TYPE>::nthTime,                      \
           (arg("self"), arg("index")),                                        \
           "returns :class:`mantid.kernel.DateAndTime`")                       \
      .def("getStatistics", &TimeSeriesProperty<TYPE>::getStatistics,          \
           arg("self"),                                                        \
           "returns :class:`mantid.kernel.TimeSeriesPropertyStatistics`")      \
      .def("timeAverageValue", &TimeSeriesProperty<TYPE>::timeAverageValue,    \
           arg("self"))                                                        \
      .def("dtype", &dtype<TYPE>, arg("self"), "returns :`std::string`");

} // namespace

void export_TimeSeriesProperty_Double() {
  EXPORT_TIMESERIES_PROP(double, Float);
}

void export_TimeSeriesProperty_Bool() { EXPORT_TIMESERIES_PROP(bool, Bool); }

void export_TimeSeriesProperty_Int32() {
  EXPORT_TIMESERIES_PROP(int32_t, Int32);
}

void export_TimeSeriesProperty_Int64() {
  EXPORT_TIMESERIES_PROP(int64_t, Int64);
}

void export_TimeSeriesProperty_String() {
  EXPORT_TIMESERIES_PROP(std::string, String);
}

void export_TimeSeriesPropertyStatistics() {
  class_<Mantid::Kernel::TimeSeriesPropertyStatistics>(
      "TimeSeriesPropertyStatistics", no_init)
      .add_property("minimum",
                    &Mantid::Kernel::TimeSeriesPropertyStatistics::minimum)
      .add_property("maximum",
                    &Mantid::Kernel::TimeSeriesPropertyStatistics::maximum)
      .add_property("mean", &Mantid::Kernel::TimeSeriesPropertyStatistics::mean)
      .add_property("median",
                    &Mantid::Kernel::TimeSeriesPropertyStatistics::median)
      .add_property(
           "standard_deviation",
           &Mantid::Kernel::TimeSeriesPropertyStatistics::standard_deviation)
      .add_property("duration",
                    &Mantid::Kernel::TimeSeriesPropertyStatistics::duration);
}
