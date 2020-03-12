# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2020 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
import collections.abc
from numbers import Integral
from typing import Iterator, Mapping, overload

import numpy as np
import AbinsModules
from AbinsModules import AbinsConstants

class KpointsData(collections.abc.Mapping):
    """Immutable container for lattice dynamics data at specific k-points.

    This object resembles a dictionary with the following form, and is
    initialised with such a dictionary as input.

    data = {"frequencies": numpy.array,
            "atomic_displacements: numpy.array,
            "weights": numpy.array,
            "k_vectors": numpy.array}


    Items in the dictionary are numpy arrays with the following formats:

    "weights" - symmetry weights of k-points; weights.shape == (num_k,);

    "k_vectors"  - values of k-points;  k_vectors.shape == (num_k, 3)

    "frequencies" - frequencies for all k-points, ordered with dimensions
        (num_k, num_freq)

    "atomic_displacements - atomic displacements for all k-points; dimensions
                           (num_k, num_atoms, num_freq, 3)

    """
    def __init__(self, data: Mapping[str, np.array]) -> None:

        self._check_content(data)
        self._check_dimensions(data)

        self._data = data

    @overload  # noqa F811
    def __getitem__(self, item: str) -> np.array:
        ...

    def __getitem__(self, item):  # noqa F811
        return self._data[item]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator:
        return iter(self._data)

    @staticmethod
    def _check_dimensions(data: Mapping[str, np.ndarray]) -> None:
        num_k = data["weights"].shape[0]
        if any([(num_k != data["k_vectors"].shape[0]),
                (num_k != data["frequencies"].shape[0]),
                (num_k != data["atomic_displacements"].shape[0])]):
            raise ValueError("Inconsistent number of k-points")

    @staticmethod
    def _check_content(items: Mapping[str, np.ndarray]) -> None:
        """Check KpointsData input (or object) for valid types/values"""

        if not isinstance(items, dict):
            raise ValueError("Input to KpointsData should be a dictionary.")

        if not sorted(items.keys()) == sorted(AbinsConstants.ALL_KEYWORDS_K_DATA):
            raise ValueError("Invalid structure of the dictionary.")

        dim = 3

        # unit_cell
        unit_cell = items["unit_cell"]
        if not (isinstance(unit_cell, np.ndarray)
                and unit_cell.shape == (dim, dim)
                and unit_cell.dtype.num == AbinsConstants.FLOAT_ID
                ):
            raise ValueError("Invalid values of unit cell vectors.")

        #  "weights"
        weights = items["weights"]

        if not (isinstance(weights, np.ndarray)
                and weights.dtype.num == AbinsConstants.FLOAT_ID
                and np.allclose(weights, weights[weights >= 0])):
            raise ValueError("Invalid value of weights.")

        #  "k_vectors"
        k_vectors = items["k_vectors"]

        if not (isinstance(k_vectors, np.ndarray)
                and k_vectors.shape[1] == dim
                and k_vectors.dtype.num == AbinsConstants.FLOAT_ID
                ):
            raise ValueError("Invalid value of k_vectors.")

        #  "frequencies"
        frequencies = items["frequencies"]
        num_freq = frequencies.shape[1]
        if not (isinstance(frequencies, np.ndarray)
                and frequencies.dtype.num == AbinsConstants.FLOAT_ID):
            raise ValueError("Invalid value of frequencies.")

        # "atomic_displacements"
        atomic_displacements = items["atomic_displacements"]
        if not (isinstance(atomic_displacements, np.ndarray)
                and atomic_displacements.shape[2:] == (num_freq, dim)
                #and atomic_displacements.shape[1] % dim == 0  # Number of frequencies should be a multiple of 3
                and atomic_displacements.dtype.num == AbinsConstants.COMPLEX_ID):
            raise ValueError("Invalid value of atomic_displacements.")

    def get_gamma_point_data(self):
        """
        Extracts k points data only for Gamma point.
        :returns: dictionary with data only for Gamma point
        """
        gamma_pkt_index = -1

        # look for index of Gamma point
        for k in self.extract()["k_vectors"]:
            if np.linalg.norm(self._data["k_vectors"][k]) < AbinsConstants.SMALL_K:
                gamma_pkt_index = k
                break
        if gamma_pkt_index == -1:
            raise ValueError("Gamma point not found.")

        gamma = AbinsConstants.GAMMA_POINT

        k_points = {"weights": {gamma: self._data["weights"][gamma_pkt_index]},
                    "k_vectors": {gamma: self._data["k_vectors"][gamma_pkt_index]},
                    "frequencies": {gamma: self._data["frequencies"][gamma_pkt_index]},
                    "atomic_displacements": {gamma: self._data["atomic_displacements"][gamma_pkt_index]}
                    }
        return k_points

    def extract(self):
        # Abins currently expects data entries to be dicts, indexed by k as string representations of integer indices
        return {key: {str(k_index): data for k_index, data in enumerate(item)} for key, item in self._data.items()}

    def __str__(self):
        return "K-points data"
