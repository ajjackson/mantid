# Pull in the version number here for Kernel/src/MantidVersion.cpp.in . This roughly follows Semantic Versioning,
# https://semver.org, but will adhere to PEP440 https://peps.python.org/pep-0440/
#
# This implementation assumes the build is run from a Git repository. and uses Python package versioningit. It is
# extracted assuming it is set in the form:
#
# Major.Minor.Patch(.Patch)(.devN|rcN)(+abc)
#
# where:
#
# * major, minor and patch numbers must be specified
# * there can be as many dot separated patch numbers as required (this accommodates nightly versions, e.g.
#   6.4.20220915.1529)
# * the tweak version number can consist of:
#
#   * a .devN version number (automatically generated by versioningit) OR a release candidate number of the form "rcN"
#   * a local version label starting with a "+" and followed by alphanumeric characters or dots.
#
# Examples:
#
# * 6.6.0 - a release
# * 6.6.0rc2 - a release candidate build
# * 6.6.20230314.1458 - a nightly build
# * 6.6.20230314.1458.dev123 - a development build 123 commits ahead of the last git tag
# * 6.6.20230314.1458.dev123+uncommitted - as above with uncommitted changes
# * 6.6.20230314.1458+ill - a nightly build with an extra local version specifier to denote useful information
# * 6.6.0+somechanges - a "local" version specifier to denote specific added features

# Use versioningit to compute version number to match conda-build
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m versioningit
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_VARIABLE _version_str
  ERROR_VARIABLE _error
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Expect format defined as in comments above. Major.Minor.Patch(.Patch)(.devN|rcN)(+abc)
if(_version_str MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+(\\.[0-9]+)*)(rc[0-9]+)?(\\.dev[0-9]+)?(\\+[A-Za-z0-9.]+)?")
  # Version components
  set(VERSION_MAJOR ${CMAKE_MATCH_1})
  set(VERSION_MINOR ${CMAKE_MATCH_2})
  set(VERSION_PATCH ${CMAKE_MATCH_3})
  # Ignore group 4 because it's contained within group 3.
  set(VERSION_TWEAK ${CMAKE_MATCH_5}${CMAKE_MATCH_6}${CMAKE_MATCH_7})
else()
  message(FATAL_ERROR "Error extracting version elements from: ${_version_str}\n${_error}")
endif()
message(STATUS "Version: ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_TWEAK}")
message(STATUS "Major: ${VERSION_MAJOR}")
message(STATUS "Minor: ${VERSION_MINOR}")
message(STATUS "Patch: ${VERSION_PATCH}")
message(STATUS "Tweak: ${VERSION_TWEAK}")
if(NOT _version_str STREQUAL "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}${VERSION_TWEAK}")
  message(FATAL_ERROR "Error when extracting version information from version string: ${_version_str}")
endif()

# Revision information
execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
  OUTPUT_VARIABLE REVISION_FULL
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
  OUTPUT_VARIABLE REVISION_SHORT
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Historically the short revision has been prefixed with a 'g'
set(REVISION_SHORT "g${REVISION_SHORT}")

# Get the date of the last commit
execute_process(
  COMMAND ${GIT_EXECUTABLE} log -1 --format=format:%cD
  OUTPUT_VARIABLE REVISION_DATE
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
string(SUBSTRING ${REVISION_DATE} 0 16 REVISION_DATE)
