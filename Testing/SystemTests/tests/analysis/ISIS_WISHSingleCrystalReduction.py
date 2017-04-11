
from mantid.simpleapi import *
import stresstesting
import numpy as np


class WISHSingleCrystalPeakPredictionTest(stresstesting.MantidStressTest):
    """
    At the time of writing WISH users rely quite heavily on the PredictPeaks 
    algorithm. As WISH has tubes rather than rectangular detectors sometimes
    peaks fall between the gaps in the tubes.

    Here we check that PredictPeaks works on a real WISH dataset & UB. This also
    includes an example of a peak whose center is predicted to fall between two
    tubes.
    """

    def requiredFiles(self):
        return ["WISH00038237.raw", "WISHPredictedSingleCrystalPeaks.nxs"]

    def requiredMemoryMB(self):
        # Need lots of memory for full WISH dataset
        return 16000

    def cleanup(self):
        pass

    def runTest(self):
        ws = LoadRaw(Filename='WISH00038237.raw', OutputWorkspace='38237')
        ws = ConvertUnits(ws, 'dSpacing', OutputWorkspace='38237')
        UB = np.array([[-0.00601763,  0.07397297,  0.05865706],
                       [ 0.05373321,  0.050198,   -0.05651455],
                       [-0.07822144,  0.0295911,  -0.04489172]])

        SetUB(ws, UB=UB)

        self._peaks = PredictPeaks(ws, WavelengthMin=0.1, WavelengthMax=100, 
                                   OutputWorkspace='peaks')
        # We specifically want to check peak -5 -1 -7 exists, so filter for it
        self._filtered = FilterPeaks(self._peaks, "h^2+k^2+l^2", 75, '=',
                                    OutputWorkspace='filtered')

    def validate(self):
        self.assertEqual(self._peaks.rowCount(), 510)
        self.assertEqual(self._filtered.rowCount(), 6)
        peak = self._filtered.row(2) 
        
        # This is an example of a peak that is known to fall between the gaps
        # in WISH tubes. Specifically check this one is predicted to exist
        # because past bugs have been found in the ray tracing
        peakMatches = peak['h'] == -5 and peak['k'] == -1 and peak['l'] == -7
        self.assertTrue(peakMatches)

        return self._peaks.name(), "WISHPredictedSingleCrystalPeaks.nxs"
