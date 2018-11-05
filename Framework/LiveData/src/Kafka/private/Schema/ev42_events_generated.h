// Mantid Repository : https://github.com/mantidproject/mantid
//
// Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
//     NScD Oak Ridge National Laboratory, European Spallation Source
//     & Institut Laue - Langevin
// SPDX - License - Identifier: GPL - 3.0 +
// clang-format off
// automatically generated by the FlatBuffers compiler, do not modify

#ifndef FLATBUFFERS_GENERATED_EV42EVENTS_H_
#define FLATBUFFERS_GENERATED_EV42EVENTS_H_

#include "flatbuffers/flatbuffers.h"

#include "is84_isis_events_generated.h"

struct ISISData;


struct EventMessage;

enum FacilityData {
  FacilityData_NONE = 0,
  FacilityData_ISISData = 1,
  FacilityData_MIN = FacilityData_NONE,
  FacilityData_MAX = FacilityData_ISISData
};

inline const char **EnumNamesFacilityData() {
  static const char *names[] = { "NONE", "ISISData", nullptr };
  return names;
}

inline const char *EnumNameFacilityData(FacilityData e) { return EnumNamesFacilityData()[static_cast<int>(e)]; }

inline bool VerifyFacilityData(flatbuffers::Verifier &verifier, const void *union_obj, FacilityData type);

struct EventMessage FLATBUFFERS_FINAL_CLASS : private flatbuffers::Table {
  enum {
    VT_SOURCE_NAME = 4,
    VT_MESSAGE_ID = 6,
    VT_PULSE_TIME = 8,
    VT_TIME_OF_FLIGHT = 10,
    VT_DETECTOR_ID = 12,
    VT_FACILITY_SPECIFIC_DATA_TYPE = 14,
    VT_FACILITY_SPECIFIC_DATA = 16
  };
  const flatbuffers::String *source_name() const { return GetPointer<const flatbuffers::String *>(VT_SOURCE_NAME); }
  uint64_t message_id() const { return GetField<uint64_t>(VT_MESSAGE_ID, 0); }
  uint64_t pulse_time() const { return GetField<uint64_t>(VT_PULSE_TIME, 0); }
  const flatbuffers::Vector<uint32_t> *time_of_flight() const { return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_TIME_OF_FLIGHT); }
  const flatbuffers::Vector<uint32_t> *detector_id() const { return GetPointer<const flatbuffers::Vector<uint32_t> *>(VT_DETECTOR_ID); }
  FacilityData facility_specific_data_type() const { return static_cast<FacilityData>(GetField<uint8_t>(VT_FACILITY_SPECIFIC_DATA_TYPE, 0)); }
  const void *facility_specific_data() const { return GetPointer<const void *>(VT_FACILITY_SPECIFIC_DATA); }
  bool Verify(flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_SOURCE_NAME) &&
           verifier.Verify(source_name()) &&
           VerifyField<uint64_t>(verifier, VT_MESSAGE_ID) &&
           VerifyField<uint64_t>(verifier, VT_PULSE_TIME) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_TIME_OF_FLIGHT) &&
           verifier.Verify(time_of_flight()) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_DETECTOR_ID) &&
           verifier.Verify(detector_id()) &&
           VerifyField<uint8_t>(verifier, VT_FACILITY_SPECIFIC_DATA_TYPE) &&
           VerifyField<flatbuffers::uoffset_t>(verifier, VT_FACILITY_SPECIFIC_DATA) &&
           VerifyFacilityData(verifier, facility_specific_data(), facility_specific_data_type()) &&
           verifier.EndTable();
  }
};

struct EventMessageBuilder {
  flatbuffers::FlatBufferBuilder &fbb_;
  flatbuffers::uoffset_t start_;
  void add_source_name(flatbuffers::Offset<flatbuffers::String> source_name) { fbb_.AddOffset(EventMessage::VT_SOURCE_NAME, source_name); }
  void add_message_id(uint64_t message_id) { fbb_.AddElement<uint64_t>(EventMessage::VT_MESSAGE_ID, message_id, 0); }
  void add_pulse_time(uint64_t pulse_time) { fbb_.AddElement<uint64_t>(EventMessage::VT_PULSE_TIME, pulse_time, 0); }
  void add_time_of_flight(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> time_of_flight) { fbb_.AddOffset(EventMessage::VT_TIME_OF_FLIGHT, time_of_flight); }
  void add_detector_id(flatbuffers::Offset<flatbuffers::Vector<uint32_t>> detector_id) { fbb_.AddOffset(EventMessage::VT_DETECTOR_ID, detector_id); }
  void add_facility_specific_data_type(FacilityData facility_specific_data_type) { fbb_.AddElement<uint8_t>(EventMessage::VT_FACILITY_SPECIFIC_DATA_TYPE, static_cast<uint8_t>(facility_specific_data_type), 0); }
  void add_facility_specific_data(flatbuffers::Offset<void> facility_specific_data) { fbb_.AddOffset(EventMessage::VT_FACILITY_SPECIFIC_DATA, facility_specific_data); }
  EventMessageBuilder(flatbuffers::FlatBufferBuilder &_fbb) : fbb_(_fbb) { start_ = fbb_.StartTable(); }
  EventMessageBuilder &operator=(const EventMessageBuilder &);
  flatbuffers::Offset<EventMessage> Finish() {
    auto o = flatbuffers::Offset<EventMessage>(fbb_.EndTable(start_, 7));
    return o;
  }
};

inline flatbuffers::Offset<EventMessage> CreateEventMessage(flatbuffers::FlatBufferBuilder &_fbb,
   flatbuffers::Offset<flatbuffers::String> source_name = 0,
   uint64_t message_id = 0,
   uint64_t pulse_time = 0,
   flatbuffers::Offset<flatbuffers::Vector<uint32_t>> time_of_flight = 0,
   flatbuffers::Offset<flatbuffers::Vector<uint32_t>> detector_id = 0,
   FacilityData facility_specific_data_type = FacilityData_NONE,
   flatbuffers::Offset<void> facility_specific_data = 0) {
  EventMessageBuilder builder_(_fbb);
  builder_.add_pulse_time(pulse_time);
  builder_.add_message_id(message_id);
  builder_.add_facility_specific_data(facility_specific_data);
  builder_.add_detector_id(detector_id);
  builder_.add_time_of_flight(time_of_flight);
  builder_.add_source_name(source_name);
  builder_.add_facility_specific_data_type(facility_specific_data_type);
  return builder_.Finish();
}

inline bool VerifyFacilityData(flatbuffers::Verifier &verifier, const void *union_obj, FacilityData type) {
  switch (type) {
    case FacilityData_NONE: return true;
    case FacilityData_ISISData: return verifier.VerifyTable(reinterpret_cast<const ISISData *>(union_obj));
    default: return false;
  }
}

inline const EventMessage *GetEventMessage(const void *buf) { return flatbuffers::GetRoot<EventMessage>(buf); }

inline bool VerifyEventMessageBuffer(flatbuffers::Verifier &verifier) { return verifier.VerifyBuffer<EventMessage>(); }

inline const char *EventMessageIdentifier() { return "ev42"; }

inline bool EventMessageBufferHasIdentifier(const void *buf) { return flatbuffers::BufferHasIdentifier(buf, EventMessageIdentifier()); }

inline void FinishEventMessageBuffer(flatbuffers::FlatBufferBuilder &fbb, flatbuffers::Offset<EventMessage> root) { fbb.Finish(root, EventMessageIdentifier()); }


#endif  // FLATBUFFERS_GENERATED_EV42EVENTS_H_
// clang-format on
