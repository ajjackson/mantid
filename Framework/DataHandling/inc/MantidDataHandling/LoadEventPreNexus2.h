// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2010 ISIS Rutherford Appleton Laboratory UKRI,
//   NScD Oak Ridge National Laboratory, European Spallation Source,
//   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
// SPDX - License - Identifier: GPL - 3.0 +
#pragma once

#include "MantidAPI/DeprecatedAlgorithm.h"
#include "MantidAPI/IFileLoader.h"
#include "MantidDataHandling/DllConfig.h"
#include "MantidDataObjects/EventWorkspace.h"
#include "MantidDataObjects/Events.h"
#include "MantidKernel/BinaryFile.h"
#include "MantidKernel/FileDescriptor.h"
#include <fstream>
#include <string>
#include <vector>

namespace Mantid {
namespace DataHandling {
/** @class Mantid::DataHandling::LoadEventPreNexus2

    A data loading routine for SNS pre-nexus event files
*/

/// This define is used to quickly turn parallel code on or off.
#undef LOADEVENTPRENEXUS_ALLOW_PARALLEL

/// Make the code clearer by having this an explicit type
using PixelType = int;

/// Type for the DAS time of flight (data file)
using DasTofType = int;

/// Structure that matches the form in the binary event list.
#pragma pack(push, 4) // Make sure the structure is 8 bytes.
struct DasEvent {
  /// Time of flight.
  DasTofType tof;
  /// Pixel identifier as published by the DAS/DAE/DAQ.
  PixelType pid;
};
#pragma pack(pop)

/// Structure used as an intermediate for parallel processing of events
#pragma pack(push, 4) // Make sure the structure is 8 bytes.
struct IntermediateEvent {
  /// Time of flight.
  DasTofType tof;
  /// Pixel identifier as published by the DAS/DAE/DAQ.
  PixelType pid;
  /// Frame index (pulse # of this event)
  size_t frame_index;
  /// Period of the event (not really used at this time)
  uint32_t period;
};
#pragma pack(pop)

/// Structure that matches the form in the new pulseid files.
#pragma pack(push, 4) // Make sure the structure is 16 bytes.
struct Pulse {
  /// The number of nanoseconds since the seconds field. This is not necessarily
  /// less than one second.
  uint32_t nanoseconds;

  /// The number of seconds since January 1, 1990.
  uint32_t seconds;

  /// The index of the first event for this pulse.
  uint64_t event_index;

  /// The proton charge for the pulse.
  double pCurrent;
};
#pragma pack(pop)

class MANTID_DATAHANDLING_DLL LoadEventPreNexus2 : public API::IFileLoader<Kernel::FileDescriptor>,
                                                   public API::DeprecatedAlgorithm {
public:
  /// Constructor
  LoadEventPreNexus2();
  /// Algorithm's name
  const std::string name() const override { return "LoadEventPreNexus"; }
  /// Algorithm's version
  int version() const override { return (2); }
  const std::vector<std::string> seeAlso() const override { return {"LoadPreNexus"}; }
  /// Algorithm's category for identification
  const std::string category() const override { return "DataHandling\\PreNexus"; }
  /// Algorithm's aliases
  const std::string alias() const override { return "LoadEventPreNeXus2"; }
  /// Summary of algorithms purpose
  const std::string summary() const override {
    return "Loads SNS raw neutron event data format and stores it in a "
           "workspace.";
  }
  /// Returns a confidence value that this algorithm can load a file
  int confidence(Kernel::FileDescriptor &descriptor) const override;

private:
  /// Initialisation code
  void init() override;
  /// Execution code
  void exec() override;

  std::unique_ptr<Mantid::API::Progress> prog = nullptr;

  DataObjects::EventWorkspace_sptr localWorkspace; //< Output EventWorkspace
  std::vector<int64_t> spectra_list;               ///< the list of Spectra

  /// The times for each pulse.
  std::vector<Types::Core::DateAndTime> pulsetimes;
  /// The index of the first event in each pulse.
  std::vector<uint64_t> event_indices;
  /// The proton charge on a pulse by pulse basis.
  std::vector<double> proton_charge;
  /// The total proton charge for the run.
  double proton_charge_tot;
  /// The value of the vector is the workspace index. The index into it is the
  /// pixel ID from DAS
  std::vector<std::size_t> pixel_to_wkspindex;
  /// Map between the DAS pixel IDs and our pixel IDs, used while loading.
  std::vector<PixelType> pixelmap;

  /// The maximum detector ID possible
  Mantid::detid_t detid_max;

  /// Handles loading from the event file
  std::unique_ptr<Mantid::Kernel::BinaryFile<DasEvent>> eventfile;
  std::size_t num_events; ///< The number of events in the file
  std::size_t num_pulses; ///< the number of pulses
  uint32_t numpixel;      ///< the number of pixels

  std::size_t num_good_events;       ///< The number of good events loaded
  std::size_t num_error_events;      ///< The number of error events encountered
  std::size_t num_bad_events;        ///< The number of bad events. Part of error
                                     ///< events
  std::size_t num_wrongdetid_events; ///< The number of events with wrong
  /// detector IDs. Part of error events.
  std::set<PixelType> wrongdetids; ///< set of all wrong detector IDs
  std::map<PixelType, size_t> wrongdetidmap;
  std::vector<std::vector<Types::Core::DateAndTime>> wrongdetid_pulsetimes;
  std::vector<std::vector<double>> wrongdetid_tofs;

  /// the number of events that were ignored (not loaded) because, e.g. of only
  /// loading some spectra.
  std::size_t num_ignored_events;
  std::size_t first_event; ///< The first event to load (count from zero)
  std::size_t max_events;  ///< Number of events to load

  /// Set to true if a valid Mapping file was provided.
  bool using_mapping_file;

  /// For loading only some spectra
  bool loadOnlySomeSpectra;
  /// Handle to the loaded spectra map
  std::map<int64_t, bool> spectraLoadMap;

  /// Longest TOF limit
  double longest_tof;
  /// Shortest TOF limit
  double shortest_tof;

  /// Flag to allow for parallel loading
  bool parallelProcessing;

  /// Whether or not the pulse times are sorted in increasing order.
  bool pulsetimesincreasing;

  /// sample environment event
  std::vector<detid_t> mSEids;
  std::map<size_t, detid_t> mSEmap;
  std::vector<std::vector<int64_t>> mSEpulseids;
  std::vector<std::vector<double>> mSEtofs;

  /// Investigation properties
  bool m_dbOutput;
  int m_dbOpBlockNumber;
  size_t m_dbOpNumEvents;
  size_t m_dbOpNumPulses;

  void loadPixelMap(const std::string &filename);

  void openEventFile(const std::string &filename);

  void readPulseidFile(const std::string &filename, const bool throwError);

  void runLoadInstrument(const std::string &eventfilename, const API::MatrixWorkspace_sptr &localWorkspace);

  inline void fixPixelId(PixelType &pixel, uint32_t &period) const;

  void procEvents(DataObjects::EventWorkspace_sptr &workspace);

  void procEventsLinear(DataObjects::EventWorkspace_sptr &workspace,
                        std::vector<Types::Event::TofEvent> **arrayOfVectors, DasEvent *event_buffer,
                        size_t current_event_buffer_size, size_t fileOffset, bool dbprint);

  void setProtonCharge(DataObjects::EventWorkspace_sptr &workspace);

  void addToWorkspaceLog(const std::string &logtitle, size_t mindex);

  void processImbedLogs();

  void debugOutput(bool doit, size_t mindex);

  void unmaskVetoEventIndex();

  API::MatrixWorkspace_sptr generateEventDistribtionWorkspace();

  void createOutputWorkspace(const std::string &event_filename);

  /// Processing the input properties for purpose of investigation
  void processInvestigationInputs();
};
} // namespace DataHandling
} // namespace Mantid
