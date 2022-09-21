# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#   NScD Oak Ridge National Laboratory, European Spallation Source,
#   Institut Laue - Langevin & CSNS, Institute of High Energy Physics, CAS
# SPDX - License - Identifier: GPL - 3.0 +
from io import BytesIO
import unittest

import numpy as np
from numpy.testing import assert_allclose

from mantid.kernel import Atom

import abins.input
from abins.input import DMOL3Loader
import abins.test_helpers


class AbinsLoadDMOL3InternalsTest(unittest.TestCase):
    """Check fragments of DMOL3 parser separately from whole file"""
    def test_read_lattice_vectors(self):
        file_without_vectors = BytesIO(b'No vectors in this file')
        unit_cell = DMOL3Loader._read_lattice_vectors(
            file_without_vectors)

        assert_allclose(unit_cell, np.zeros((3, 3)))

        file_with_vectors = BytesIO(b"""
INCOOR, atomic coordinates in au (for archive):
______________________________________________________________________>8

$cell vectors
             16.74108374131821    0.00000000000000    0.00000000000000
             -8.37054187065910   14.49820380686420    0.00000000000000
              0.00000000000000    0.00000000000000    9.52044021771771
$coordinates
F             2.22656413759532   -1.33383475061772   -1.80983568538814
F             2.42243481736875    8.68442407988897    6.67097246055480
F            -2.34542583215868   10.72867081788549    2.94943237944894
""")
        unit_cell = DMOL3Loader._read_lattice_vectors(file_with_vectors)

        assert_allclose(unit_cell,
                        np.array([[ 8.859     ,  0.        ,  0.        ],
                                  [-4.4295    ,  7.67211905,  0.        ],
                                  [ 0.        ,  0.        ,  5.038     ]]))

    def test_read_positions(self):
        file_with_coords_end = BytesIO(b"""
             -8.37054187065910   14.49820380686420    0.00000000000000
              0.00000000000000    0.00000000000000    9.52044021771771
$coordinates
F             2.22656413759532   -1.33383475061772   -1.80983568538814
F             2.42243481736875    8.68442407988897    6.67097246055480
Na           -2.34542583215868   10.72867081788549    2.94943237944894
Si            0.04185270935330    2.59517848080392   -1.80983568538814
$end
DUMMY CONTENT
""")
        atoms = DMOL3Loader._read_atomic_coordinates(file_with_coords_end)

        assert_allclose(
            [atom['coord'] for _, atom in atoms.items()],
            np.array([[2.22656413759532, -1.33383475061772, -1.80983568538814],
                      [2.42243481736875, 8.68442407988897, 6.67097246055480],
                      [-2.34542583215868, 10.72867081788549, 2.94943237944894],
                      [0.04185270935330, 2.59517848080392, -1.80983568538814]]
                     ) * 0.52917721067  # Convert Bohr to Angstrom
        )

        self.assertEqual(['F', 'F', 'Na', 'Si'],
                         [atom['symbol'] for _, atom in atoms.items()])

        # At this point the atomic mass should still be the Mantid value
        self.assertEqual(atoms['atom_0']['mass'], Atom(symbol='F').mass)

        self.assertEqual(list(range(len(atoms))),
                         [atom['sort'] for _, atom in atoms.items()])

    def test_read_masses(self):

        # usual case: break on Molecular Mass:
        file_with_masses = BytesIO(b"""
    z     -0.5431      -0.4170      -0.0002      -0.0001      -0.3565      -0.0000       0.0000       0.0000       0.0000

 STANDARD THERMODYNAMIC QUANTITIES AT   298.15 K  AND     1.00 ATM

   Zero point vibrational energy:       38.528 kcal/mol

   Atom    1 Element F  Has Mass   18.50000
   Atom    2 Element F  Has Mass   18.50000
   Atom    3 Element F  Has Mass   21.00000
   Atom    4 Element F  Has Mass   22.00000
   Molecular Mass:   564.166320 amu
   Principal axes and moments of inertia in atomic units:
                               1              2              3
    Eigenvalues --        8975.59891    26615.40273    29256.87074
          X                  0.82797        0.50000        0.25389
""")

        # If not found, parser should still break cleanly at file end
        file_with_masses_noend = BytesIO(b"""
    z     -0.5431      -0.4170      -0.0002      -0.0001      -0.3565      -0.0000       0.0000       0.0000       0.0000

 STANDARD THERMODYNAMIC QUANTITIES AT   298.15 K  AND     1.00 ATM

   Zero point vibrational energy:       38.528 kcal/mol

   Atom    1 Element F  Has Mass   18.50000
   Atom    2 Element F  Has Mass   18.50000
   Atom    3 Element F  Has Mass   21.00000
   Atom    4 Element F  Has Mass   22.00000
   Principal axes and moments of inertia in atomic units:
                               1              2              3
    Eigenvalues --        8975.59891    26615.40273    29256.87074
          X                  0.82797        0.50000        0.25389
""")

        for file_stream in (file_with_masses, file_with_masses_noend):
            masses = DMOL3Loader._read_masses_from_file(file_stream)
            assert_allclose(masses, [18.5, 18.5, 21., 22.])


class AbinsLoadDMOL3Test(unittest.TestCase, abins.input.Tester):
    """Check entire input files against expected outputs"""
    def tearDown(self):
        abins.test_helpers.remove_output_files(list_of_names=["_LoadDMOL3"])

    _gamma_dmol3 = "LTA_40_O2_LoadDMOL3"
    _gamma_no_h_dmol3 = "Na2SiF6_LoadDMOL3"
    _molecule_dmol3 = "methane_LoadDMOL3"
    _molecule_opt_dmol3 = "methane_opt_LoadDMOL3"

    def test_gamma_dmol3(self):
        self.check(name=self._gamma_dmol3, loader=DMOL3Loader)
        self.check(name=self._gamma_no_h_dmol3, loader=DMOL3Loader)

    def test_molecule_dmol3(self):
        self.check(name=self._molecule_dmol3, loader=DMOL3Loader)
        self.check(name=self._molecule_opt_dmol3, loader=DMOL3Loader)


if __name__ == '__main__':
    unittest.main()
