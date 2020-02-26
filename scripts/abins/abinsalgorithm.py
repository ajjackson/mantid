# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +

# Supporting functions for the Abins Algorithm that don't belong in
# another part of AbinsModules.
import os

from mantid.api import mtd, FileAction, FileProperty, WorkspaceProperty
from mantid.kernel import Direction, StringListValidator, StringArrayProperty, logger

import abins
from abins.constants import ALL_INSTRUMENTS, ALL_SAMPLE_FORMS
from abins.instruments import get_instrument


class AbinsAlgorithm:
    """Class providing shared utility for multiple inheritence by 1D, 2D implementations"""

    def _get_common_properties(self) -> None:
        """From user input, set properties common to Abins 1D and 2D versions"""
        self._ab_initio_program = self.getProperty("AbInitioProgram").value
        self._vibrational_or_phonon_data_file = self.getProperty("VibrationalOrPhononFile").value
        self._out_ws_name = self.getPropertyValue('OutputWorkspace')

        self._temperature = self.getProperty("TemperatureInKelvin").value
        self._bin_width = self.getProperty("BinWidthInWavenumber").value
        self._sample_form = self.getProperty("SampleForm").value

        self._atoms = self.getProperty("Atoms").value
        self._sum_contributions = self.getProperty("SumContributions").value
        self._scale_by_cross_section = self.getPropertyValue('ScaleByCrossSection')

        # conversion from str to int
        self._num_quantum_order_events = int(self.getProperty("QuantumOrderEventsNumber").value)

        instrument_name = self.getProperty("Instrument").value
        if instrument_name in ALL_INSTRUMENTS:
            self._instrument_name = instrument_name
            self._instrument = get_instrument(self._instrument_name)
        else:
            raise ValueError("Unknown instrument %s" % instrument_name)

    def declare_common_properties(self) -> None:
        """Declare properties common to Abins 1D and 2D versions"""
        self.declareProperty(FileProperty("VibrationalOrPhononFile", "",
                                          action=FileAction.Load,
                                          direction=Direction.Input,
                                          extensions=["phonon", "out", "outmol", "log", "LOG"]),
                             doc="File with the data from a vibrational or phonon calculation.")

        self.declareProperty(name="AbInitioProgram",
                             direction=Direction.Input,
                             defaultValue="CASTEP",
                             validator=StringListValidator(["CASTEP", "CRYSTAL", "DMOL3", "GAUSSIAN"]),
                             doc="An ab initio program which was used for vibrational or phonon calculation.")

        self.declareProperty(WorkspaceProperty("OutputWorkspace", '', Direction.Output),
                             doc="Name to give the output workspace.")

        self.declareProperty(name="TemperatureInKelvin",
                             direction=Direction.Input,
                             defaultValue=10.0,
                             doc="Temperature in K for which dynamical structure factor S should be calculated.")

        self.declareProperty(name="BinWidthInWavenumber", defaultValue=1.0, doc="Width of bins used during rebining.")

        self.declareProperty(name="SampleForm",
                             direction=Direction.Input,
                             defaultValue="Powder",
                             validator=StringListValidator(ALL_SAMPLE_FORMS),
                             # doc="Form of the sample: SingleCrystal or Powder.")
                             doc="Form of the sample: Powder.")

        self.declareProperty(StringArrayProperty("Atoms", Direction.Input),
                             doc="List of atoms to use to calculate partial S."
                                 "If left blank, workspaces with S for all types of atoms will be calculated. "
                                 "Element symbols will be interpreted as a sum of all atoms of that element in the "
                                 "cell. 'atomN' or 'atom_N' (where N is a positive integer) will be interpreted as "
                                 "individual atoms, indexing from 1 following the order of the input data.")

        self.declareProperty(name="SumContributions", defaultValue=False,
                             doc="Sum the partial dynamical structure factors into a single workspace.")

        self.declareProperty(name="ScaleByCrossSection", defaultValue='Incoherent',
                             validator=StringListValidator(['Total', 'Incoherent', 'Coherent']),
                             doc="Scale the partial dynamical structure factors by the scattering cross section.")

        # Abins is supposed to support excitations up to fourth-order. Order 3 and 4 are currently disabled while the
        # weighting is being investigated; these intensities were unreasonably large in hydrogenous test cases
        self.declareProperty(name="QuantumOrderEventsNumber", defaultValue='1',
                             validator=StringListValidator(['1', '2']),
                             doc="Number of quantum order effects included in the calculation "
                                 "(1 -> FUNDAMENTALS, 2-> first overtone + FUNDAMENTALS + 2nd order combinations")

    def validate_common_inputs(self, issues: dict = None) -> dict:
        """Validate inputs common to Abins 1D and 2D versions

        Args:
            abins: Algorithm instance for validation
            issues: Collection of validation issues to append to. (This will be mutated without copying.)

        Returns:
            issues dict including any new issues
        """
        if issues is None:
            issues = {}

        input_file_validators = {"CASTEP": self._validate_castep_input_file,
                                 "CRYSTAL": self._validate_crystal_input_file,
                                 "DMOL3": self._validate_dmol3_input_file,
                                 "GAUSSIAN": self._validate_gaussian_input_file}
        ab_initio_program = self.getProperty("AbInitioProgram").value
        vibrational_or_phonon_data_filename = self.getProperty("VibrationalOrPhononFile").value

        output = input_file_validators[ab_initio_program](vibrational_or_phonon_data_filename)
        if output["Invalid"]:
            issues["VibrationalOrPhononFile"] = output["Comment"]

        workspace_name = self.getPropertyValue("OutputWorkspace")
        # list of special keywords which cannot be used in the name of workspace
        forbidden_keywords = ["total"]
        if workspace_name in mtd:
            issues["OutputWorkspace"] = "Workspace with name " + workspace_name + " already in use; please give " \
                                                                                  "a different name for workspace."
        elif workspace_name == "":
            issues["OutputWorkspace"] = "Please specify name of workspace."
        for word in forbidden_keywords:

            if word in workspace_name:
                issues["OutputWorkspace"] = "Keyword: " + word + " cannot be used in the name of workspace."
                break

        temperature = self.getProperty("TemperatureInKelvin").value
        if temperature < 0:
            issues["TemperatureInKelvin"] = "Temperature must be positive."

        bin_width = self.getProperty("BinWidthInWavenumber").value
        if not (isinstance(bin_width, float) and 1.0 <= bin_width <= 10.0):
            issues["BinWidthInWavenumber"] = ["Invalid bin width. Valid range is [1.0, 10.0] cm^-1"]

        return issues

    @staticmethod
    def _validate_ab_initio_file_extension(*,
                                           ab_initio_program: str,
                                           filename_full_path: str,
                                           expected_file_extension: str) -> dict:
        """
        Checks consistency between name of ab initio program and extension.
        :param expected_file_extension: file extension
        :returns: dictionary with error message
        """
        msg_err = "Invalid %s file. " % filename_full_path
        msg_rename = "Please rename your file and try again."

        # check  extension of a file
        found_filename_ext = os.path.splitext(filename_full_path)[1]
        if found_filename_ext.lower() != expected_file_extension:
            comment = "{}Output from ab initio program {} is expected." \
                      " The expected extension of file is .{}. Found: {}.{}".format(
                          msg_err, ab_initio_program, expected_file_extension, found_filename_ext, msg_rename)
            return dict(Invalid=True, Comment=comment)
        else:
            return dict(Invalid=False, Comment="")

    @classmethod
    def _validate_dmol3_input_file(cls, filename_full_path: str) -> dict:
        """
        Method to validate input file for DMOL3 ab initio program.
        :param filename_full_path: full path of a file to check.
        :returns: True if file is valid otherwise false.
        """
        logger.information("Validate DMOL3 file with vibrational data.")
        return cls._validate_ab_initio_file_extension(ab_initio_program="DMOL3",
                                                      filename_full_path=filename_full_path,
                                                      expected_file_extension=".outmol")

    @classmethod
    def _validate_gaussian_input_file(cls, filename_full_path: str) -> dict:
        """
        Method to validate input file for GAUSSIAN ab initio program.
        :param filename_full_path: full path of a file to check.
        :returns: True if file is valid otherwise false.
        """
        logger.information("Validate GAUSSIAN file with vibration data.")
        return cls._validate_ab_initio_file_extension(ab_initio_program="GAUSSIAN",
                                                      filename_full_path=filename_full_path,
                                                      expected_file_extension=".log")

    @classmethod
    def _validate_crystal_input_file(cls,
                                     filename_full_path: str) -> dict:
        """
        Method to validate input file for CRYSTAL ab initio program.
        :param filename_full_path: full path of a file to check.
        :returns: True if file is valid otherwise false.
        """
        logger.information("Validate CRYSTAL file with vibrational or phonon data.")
        return cls._validate_ab_initio_file_extension(ab_initio_program="CRYSTAL",
                                                      filename_full_path=filename_full_path,
                                                      expected_file_extension=".out")

    @classmethod
    def _validate_castep_input_file(cls, filename_full_path: str) -> dict:
        """
        Check if ab initio input vibrational or phonon file has been produced by CASTEP. Currently the crucial
        keywords in the first few lines are checked (to be modified if a better validation is found...)
        :param filename_full_path: full path of a file to check
        :returns: Dictionary with two entries "Invalid", "Comment". Valid key can have two values: True/ False. As it
                  comes to "Comment" it is an empty string if Valid:True, otherwise stores description of the problem.
        """

        logger.information("Validate CASTEP file with vibrational or phonon data.")
        msg_err = "Invalid %s file. " % filename_full_path
        output = cls._validate_ab_initio_file_extension(ab_initio_program="CASTEP",
                                                        filename_full_path=filename_full_path,
                                                        expected_file_extension=".phonon")
        if output["Invalid"]:
            return output
        else:
            # check a structure of the header part of file.
            # Here fortran convention is followed: case of letter does not matter
            with open(filename_full_path) as castep_file:
                line = cls._get_one_line(castep_file)
                if not cls._compare_one_line(line, "beginheader"):  # first line is BEGIN header
                    return dict(Invalid=True, Comment=msg_err + "The first line should be 'BEGIN header'.")

                line = cls._get_one_line(castep_file)
                if not cls._compare_one_line(one_line=line, pattern="numberofions"):
                    return dict(Invalid=True, Comment=msg_err + "The second line should include 'Number of ions'.")

                line = cls._get_one_line(castep_file)
                if not cls._compare_one_line(one_line=line, pattern="numberofbranches"):
                    return dict(Invalid=True, Comment=msg_err + "The third line should include 'Number of branches'.")

                line = cls._get_one_line(castep_file)
                if not cls._compare_one_line(one_line=line, pattern="numberofwavevectors"):
                    return dict(Invalid=True, Comment=msg_err + "The fourth line should include 'Number of wavevectors'.")

                line = cls._get_one_line(castep_file)
                if not cls._compare_one_line(one_line=line,
                                             pattern="frequenciesin"):
                    return dict(Invalid=True, Comment=msg_err + "The fifth line should be 'Frequencies in'.")

                return dict(Invalid=False, Comment="")

    @staticmethod
    def _get_one_line(file_obj=None):
        """

        :param file_obj:  file object from which reading is done
        :returns: string containing one non empty line
        """
        line = file_obj.readline().replace(" ", "").lower()

        while line and line == "":
            line = file_obj.readline().replace(" ", "").lower()

        return line

    @staticmethod
    def _compare_one_line(one_line, pattern):
        """
        compares line in the the form of string with a pattern.
        :param one_line:  line in the for mof string to be compared
        :param pattern: string which should be present in the line after removing white spaces and setting all
                        letters to lower case
        :returns:  True is pattern present in the line, otherwise False
        """
        return one_line and pattern in one_line.replace(" ", "")

    @classmethod
    def _check_common_advanced_parameters(cls, message=""):
        cls._check_general_resolution(message)
        cls._check_folder_names(message)
        cls._check_rebinning(message)
        cls._check_threshold(message)
        cls._check_chunk_size(message)
        cls._check_threads(message)

    def _check_general_resolution(self, message_end=None):
        """
        Checks general parameters used in construction resolution functions.
        :param message_end: closing part of the error message.
        """
        # check fwhm
        fwhm = abins.parameters.instruments['fwhm']
        if not (isinstance(fwhm, float) and 0.0 < fwhm < 10.0):
            raise RuntimeError("Invalid value of fwhm" + message_end)

    @staticmethod
    def _check_folder_names(message_end=None):
        """
        Checks folders names.
        :param message_end: closing part of the error message.
        """
        folder_names = []
        ab_initio_group = abins.parameters.hdf_groups['ab_initio_data']
        if not isinstance(ab_initio_group, str) or ab_initio_group == "":
            raise RuntimeError("Invalid name for folder in which the ab initio data should be stored.")
        folder_names.append(ab_initio_group)

        powder_data_group = abins.parameters.hdf_groups['powder_data']
        if not isinstance(powder_data_group, str) or powder_data_group == "":
            raise RuntimeError("Invalid value of powder_data_group" + message_end)
        elif powder_data_group in folder_names:
            raise RuntimeError("Name for powder_data_group  already used by as name of another folder.")
        folder_names.append(powder_data_group)

        crystal_data_group = abins.parameters.hdf_groups['crystal_data']
        if not isinstance(crystal_data_group, str) or crystal_data_group == "":
            raise RuntimeError("Invalid value of crystal_data_group" + message_end)
        elif crystal_data_group in folder_names:
            raise RuntimeError("Name for crystal_data_group already used as a name of another folder.")

        s_data_group = abins.parameters.hdf_groups['s_data']
        if not isinstance(s_data_group, str) or s_data_group == "":
            raise RuntimeError("Invalid value of s_data_group" + message_end)
        elif s_data_group in folder_names:
            raise RuntimeError("Name for s_data_group already used as a name of another folder.")

    @staticmethod
    def _check_rebinning(message_end=None):
        """
        Checks rebinning parameters.
        :param message_end: closing part of the error message.
        """
        min_wavenumber = abins.parameters.sampling['min_wavenumber']
        if not (isinstance(min_wavenumber, float) and min_wavenumber >= 0.0):
            raise RuntimeError("Invalid value of min_wavenumber" + message_end)

        max_wavenumber = abins.parameters.sampling['max_wavenumber']
        if not (isinstance(max_wavenumber, float) and max_wavenumber > 0.0):
            raise RuntimeError("Invalid number of max_wavenumber" + message_end)

        if min_wavenumber > max_wavenumber:
            raise RuntimeError("Invalid energy window for rebinning.")

    @staticmethod
    def _check_threshold(message_end=None):
        """
        Checks threshold for frequencies.
        :param message_end: closing part of the error message.
        """
        from abins.parameters import sampling

        freq_threshold = sampling['frequencies_threshold']
        if not (isinstance(freq_threshold, float) and freq_threshold >= 0.0):
            raise RuntimeError("Invalid value of frequencies_threshold" + message_end)

        # check s threshold
        s_absolute_threshold = sampling['s_absolute_threshold']
        if not (isinstance(s_absolute_threshold, float) and s_absolute_threshold > 0.0):
            raise RuntimeError("Invalid value of s_absolute_threshold" + message_end)

        s_relative_threshold = sampling['s_relative_threshold']
        if not (isinstance(s_relative_threshold, float) and s_relative_threshold > 0.0):
            raise RuntimeError("Invalid value of s_relative_threshold" + message_end)

    @staticmethod
    def _check_chunk_size(message_end=None):
        """
        Check optimal size of chunk
        :param message_end: closing part of the error message.
        """
        optimal_size = abins.parameters.performance['optimal_size']
        if not (isinstance(optimal_size, int) and optimal_size > 0):
            raise RuntimeError("Invalid value of optimal_size" + message_end)

    @staticmethod
    def _check_threads(message_end=None):
        """
        Checks number of threads
        :param message_end: closing part of the error message.
        """
        try:
            import pathos.multiprocessing as mp

            threads = abins.parameters.performance['threads']
            if not (isinstance(threads, int) and 1 <= threads <= mp.cpu_count()):
                raise RuntimeError("Invalid number of threads for parallelisation over atoms" + message_end)

        except ImportError:
            pass