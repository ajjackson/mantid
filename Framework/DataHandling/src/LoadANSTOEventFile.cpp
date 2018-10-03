
#include "MantidDataHandling/LoadANSTOEventFile.h"

namespace DataHandling {
namespace ANSTO {

#define EVENTFILEHEADER_BASE_MAGIC_NUMBER   0x0DAE0DAE
#define EVENTFILEHEADER_BASE_FORMAT_NUMBER  0x00010002

	// all events contain some or all of these fields
#define NVAL 5 // x, y, v, w, wa

#define SELECTION_NONE     0
#define SELECTION_INCLUDED 1
#define SELECTION_EXCLUDED 2

#pragma pack(push, 1) // otherwise may get 8 byte aligned, no good for us

	struct EventFileHeader_Base { // total content should be 16*int (64 bytes)
		int magic_number;       // must equal EVENTFILEHEADER_BASE_MAGIC_NUMBER (DAE data)
		int format_number;      // must equal EVENTFILEHEADER_BASE_FORMAT_NUMBER, identifies this header format
		int anstohm_version;    // ANSTOHM_VERSION server/filler version number that generated the file
		int pack_format;        // typically 0 if packed binary, 1 if unpacked binary.
		int oob_enabled;        // if set, OOB events can be present in the data, otherwise only neutron and t0 events are stored
		int clock_scale;        // the CLOCK_SCALE setting, ns per timestamp unit
		int spares[16 - 6];     // spares (padding)
	};

	struct EventFileHeader_Packed { // total content should be 16*int (64 bytes)
		int evt_stg_nbits_x;    // number of bits in x datum
		int evt_stg_nbits_y;    // number of bits in y datum
		int evt_stg_nbits_v;    // number of bits in v datum
		int evt_stg_nbits_w;    // number of bits in w datum
		int evt_stg_nbits_wa;   // number of bits in wa datum // MJL added 5/15 for format 0x00010002
		int evt_stg_xy_signed;  // 0 if x and y are unsigned, 1 if x and y are signed ints
		int spares[16 - 6];     // spares (padding)
	};

#pragma pack(pop)

	// event decoding state machine
	enum event_decode_state {
		// for all events
		DECODE_START,           // initial state - then DECODE_VAL_BITFIELDS (for neutron events) or DECODE_OOB_BYTE_1 (for OOB events)
								// for OOB events only
								DECODE_OOB_BYTE_1,
								DECODE_OOB_BYTE_2,
								// for all events
								DECODE_VAL_BITFIELDS,
								DECODE_DT               // final state - then output data and return to DECODE_START
	};

	/*

	// Types of OOB events, and 'NEUTRON' event.  Not all are used for all instruments, or supported yet.
	// NEUTRON = 0 = a neutron detected, FRAME_START = -2 = T0 pulse (e.g. from chopper, or from Doppler on Emu).  For most instruments, these are the only types used.
	// FRAME_AUX_START = -3 (e.g. from reflecting chopper on Emu), VETO = -6 (e.g. veto signal from ancillary)
	// BEAM_MONITOR = -7 (e.g. if beam monitors connected direct to Mesytec MCPD8 DAE)
	// RAW = -8 = pass-through, non-decoded raw event directly from the DAE (e.g. Mesytec MCPD8).  Used to access special features of DAE.
	// Other types are not used in general (DATASIZES = -1 TBD in future, FLUSH = -4 deprecated, FRAME_DEASSERT = -5 only on Fastcomtec P7888 DAE).
	const char *oob_event_type_str[16] = { // note the c values are negative, hence the reversed order...
	// NEUTRON = 0, 1-7 unused
	"NEUTRON", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN",
	// RAW = -8, BEAM_MONITOR = -7, VETO = -6, FRAME_DEASSERT = -5, FLUSH = -4, FRAME_AUX_START = -3, FRAME_START = -2, DATASIZES = -1
	"RAW", "BEAM_MONITOR", "VETO", "FRAME_DEASSERT", "FLUSH", "FRAME_AUX_START", "FRAME_START", "DATASIZES"
	};

	*/

	void ReadEventFile(IReader& loader, IEventHandler& handler, TProgressCallback* cbp_progress, bool use_tx_chopper)
	{
		// read file headers (base header then packed-format header)
		EventFileHeader_Base hdr_base;
		if (!loader.read(reinterpret_cast<char*>(&hdr_base), sizeof(hdr_base)))
			throw std::exception("unable to load EventFileHeader-Base");

		EventFileHeader_Packed hdr_packed;
		if (!loader.read(reinterpret_cast<char*>(&hdr_packed), sizeof(hdr_packed)))
			throw std::exception("unable to load EventFileHeader-Packed");

		// check header parameters
		/*
		printf(
		"magic_number=0x%08X, format_number=0x%08X, anstohm_version=0x%08X, oob_enabled=%d, clock_scale=%d ns/unit\n",
		hdr_base.magic_number,
		hdr_base.format_number,
		hdr_base.anstohm_version,
		hdr_base.oob_enabled,
		hdr_base.clock_scale);
		*/

		if (hdr_base.magic_number != EVENTFILEHEADER_BASE_MAGIC_NUMBER)
			throw std::exception("bad magic number");

		if (hdr_base.format_number > EVENTFILEHEADER_BASE_FORMAT_NUMBER) {
			char txtBuffer[255] = {};
			snprintf(
				txtBuffer,
				sizeof(txtBuffer),
				"invalid file (only format_number=%08Xh or lower)",
				EVENTFILEHEADER_BASE_FORMAT_NUMBER);
			throw std::exception(txtBuffer);
		}

		if (hdr_base.pack_format != 0)
			throw std::exception("only packed binary format is supported");

		if (hdr_base.clock_scale == 0)
			throw std::exception("clock scale cannot be zero");

		// note: in the old format 0x00010001, the evt_stg_nbits_wa did not exist and it contained evt_stg_xy_signed
		if (hdr_base.format_number <= 0x00010001) {
			hdr_packed.evt_stg_xy_signed = hdr_packed.evt_stg_nbits_wa;
			hdr_packed.evt_stg_nbits_wa = 0;
		}

		int64_t total_time = 0;
		int64_t primary_time = 0;
		int64_t auxillary_time = 0;

		// main loop
		unsigned int x = 0, y = 0, v = 0, w = 0, wa = 0; // storage for event data fields
		unsigned int *ptr_val[NVAL] = { &x, &y, &v, &w, &wa }; // used to store data into fields

																// All events are also timestamped.  The differential timestamp dt stored in each event is summed to recover the event timestamp t.
																// All timestamps are frame-relative, i.e. FRAME_START event represents T0 (e.g. from a chopper) and t is reset to 0.
																// In OOB mode and for certain DAE types only (e.g. Mesytec MCPD8), the FRAME_START event is timestamped relative to the last FRAME_START.
																// The timestamp t on the FRAME_START event is therefore the total frame duration, and this can be used to recover the absolute timestamp
																// of all events in the DAQ, if desired (e.g. for accurate timing during long term kinematic experiments).
		int dt; // , t = 0 dt may be negative occasionally for some DAE types, therefore dt and t are signed ints.

		int nbits_val_oob[NVAL] = {};

		int nbits_val_neutron[NVAL] = {
			hdr_packed.evt_stg_nbits_x,
			hdr_packed.evt_stg_nbits_y,
			hdr_packed.evt_stg_nbits_v,
			hdr_packed.evt_stg_nbits_w,
			hdr_packed.evt_stg_nbits_wa
		};

		int ind_val = 0;
		int nbits_val = 0;
		int nbits_val_filled = 0;
		int nbits_dt_filled = 0;
		int nbits_ch_used;

		int oob_en = hdr_base.oob_enabled; // will be 1 if we are reading a new OOB event file (format 0x00010002 only).
		int oob_event = 0, c = 0; // For neutron events, oob_event = 0, and for OOB events, oob_event = 1 and c indicates the OOB event type. c<0 for all OOB events currently.

		event_decode_state state = DECODE_START; // event decoding state machine
		bool event_ended = false;
		bool _cancel = false;

		while (true) {

			// handle interrupts
			if (_cancel) {
				return;
			}

			// read next byte
			unsigned char ch;
			if (!loader.read(reinterpret_cast<char*>(&ch), 1))
				break;

			nbits_ch_used = 0; // no bits used initially, 8 to go

								// start of event processing
			if (state == DECODE_START) {

				// if OOB event mode is enabled, the leading Bit 0 of the first byte indicates whether the event is a neutron event or an OOB event
				if (!oob_en)
					state = DECODE_VAL_BITFIELDS;
				else {
					oob_event = (ch & 1);
					nbits_ch_used = 1; // leading bit used as OOB bit

					if (!oob_event)
						state = DECODE_VAL_BITFIELDS;
					else
						state = DECODE_OOB_BYTE_1;
				}

				// setup to decode new event bitfields (for both neutron and OOB events)
				for (ind_val = 0; ind_val < NVAL; ind_val++)
					*ptr_val[ind_val] = 0;

				ind_val = 0;
				nbits_val_filled = 0;

				dt = 0;
				nbits_dt_filled = 0;
			}

			// state machine for event decoding
			switch (state) {
			case DECODE_OOB_BYTE_1: // first OOB header byte
									// OOB event Byte 1:  Bit 0 = 1 = OOB event, Bit 1 = mode (only mode=0 suported currently),
									// Bits 2-5 = c (OOB event type), Bits 6-7 = bitfieldsize_x / 8.
									// bitfieldsize_x and following 2-bit bitfieldsizes are the number of bytes used to store the OOB parameter.
									// All of x,y,v,w,wa are short integers (16 bits maximum) and so bitfieldsizes = 0, 1 or 2 only.
				c = (ch >> 2) & 0xF; // Bits 2-5 = c

				if (c & 0x8)
					c |= 0xFFFFFFF0; // c is a signed parameter so sign extend - OOB events are negative values
				nbits_val_oob[0] = (ch & 0xC0) >> 3; // Bits 6-7 * 8 = bitfieldsize_x

				state = DECODE_OOB_BYTE_2; // Proceed to process second OOB event header byte next time
				break;

			case DECODE_OOB_BYTE_2: // second OOB header byte
									// bitfieldsizes for y, v, w and wa, as for bitfieldsize_x above.
				nbits_val_oob[1] = (ch & 0x03) << 3; // Bits 0-1 * 8 = bitfieldsize_y
				nbits_val_oob[2] = (ch & 0x0C) << 1; // Bits 2-3 * 8 = bitfieldsize_v
				nbits_val_oob[3] = (ch & 0x30) >> 1; // Bits 4-5 * 8 = bitfieldsize_w
				nbits_val_oob[4] = (ch & 0xC0) >> 3; // Bits 6-7 * 8 = bitfieldsize_wa

				state = DECODE_VAL_BITFIELDS; // Proceed to read and store x,y,v,w,wa for the OOB event
				break;

			case DECODE_VAL_BITFIELDS:
				// fill bits of the incoming ch to the event's bitfields.
				// stop when we've filled them all, or all bits of ch are used.
				do {
					nbits_val = (oob_event ? nbits_val_oob[ind_val] : nbits_val_neutron[ind_val]);
					if (!nbits_val) {
						nbits_val_filled = 0;
						ind_val++;
					}
					else {
						int nbits_val_to_fill = (nbits_val - nbits_val_filled);
						if ((8 - nbits_ch_used) >= nbits_val_to_fill) {
							*ptr_val[ind_val] |= ((ch >> nbits_ch_used)&((1 << nbits_val_to_fill) - 1)) << nbits_val_filled;
							nbits_val_filled = 0;
							nbits_ch_used += nbits_val_to_fill;
							ind_val++;
						}
						else {
							*ptr_val[ind_val] |= (ch >> nbits_ch_used) << nbits_val_filled;
							nbits_val_filled += (8 - nbits_ch_used);
							nbits_ch_used = 8;
						}
					}
				} while ((ind_val < NVAL) && (nbits_ch_used < 8));

				//
				if (ind_val == NVAL)
					state = DECODE_DT; // and fall through for dt processing

				if (nbits_ch_used == 8) // read next byte
					break;

			case DECODE_DT:
				if ((8 - nbits_ch_used) <= 2) {
					dt |= (ch >> nbits_ch_used) << nbits_dt_filled;
					nbits_dt_filled += (8 - nbits_ch_used);
				}
				else if ((ch & 0xC0) == 0xC0) {
					dt |= ((ch & 0x3F) >> nbits_ch_used) << nbits_dt_filled;
					nbits_dt_filled += (6 - nbits_ch_used);
				}
				else {
					dt |= (ch >> nbits_ch_used) << nbits_dt_filled;
					nbits_dt_filled += (8 - nbits_ch_used);
					event_ended = true;
				}

				break;
			}

			if (event_ended) {
				state = DECODE_START; // start on new event next time

										// update times
				total_time += dt;
				primary_time += dt;
				auxillary_time += dt;

				// is this event a frame_start? // FRAME_START is an OOB event when oob mode enabled
				bool frame_start_event = (oob_en ? (oob_event && c == -2) : (x == 0 && y == 0 && dt == 0xFFFFFFFF));

				if (oob_en || !frame_start_event) {
					if (oob_event) {
						if (c == -3) { // FRAME_AUX_START = -3
							if (!use_tx_chopper && x == 0) // 0 is the reflecting chopper and 1 is the transmission chopper
								auxillary_time = 0;
							if (use_tx_chopper && x == 1)
								auxillary_time = 0;
						}
					}
					else {
						// pass the event trhough the call back, time units in nsec
						handler.addEvent(x, y, static_cast<int64_t>(primary_time * hdr_base.clock_scale), 
											static_cast<int64_t>(auxillary_time * hdr_base.clock_scale),
											static_cast<int64_t>(total_time * hdr_base.clock_scale));
					}
				}

				if (frame_start_event) {
					// reset timestamp at start of ToF frame
					primary_time = 0;
				}

				if (cbp_progress != nullptr)
					(*cbp_progress)(loader.position());

				event_ended = false;
			}
		}
	}
}
}
