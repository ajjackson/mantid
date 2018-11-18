# Mantid Repository : https://github.com/mantidproject/mantid
#
# Copyright &copy; 2018 ISIS Rutherford Appleton Laboratory UKRI,
#     NScD Oak Ridge National Laboratory, European Spallation Source
#     & Institut Laue - Langevin
# SPDX - License - Identifier: GPL - 3.0 +
"""
    Defines a set of aliases to make accessing certain objects easier
"""
from __future__ import (absolute_import, division,
                        print_function)

from ._api import (FrameworkManagerImpl, AnalysisDataServiceImpl,
                   AlgorithmFactoryImpl, AlgorithmManagerImpl,
                   FileFinderImpl, FileLoaderRegistryImpl, FunctionFactoryImpl,
                   WorkspaceFactoryImpl, CatalogManagerImpl)
from ..kernel._aliases import lazy_instance_access

# Historically the singleton aliases mapped to the instances rather than
# the class types, i.e. AnalysisDataService is the instance and not the type,
# which doesn't match the C++ behaviour.
#
# Exit handlers are important in some cases as the associated singleton
# stores references to python objects that need to be cleaned up
# Without a python-based exit handler the singletons are only cleaned
# up after main() and this is too late to acquire the GIL to be able to
# delete the python objects.
# If you see a segfault late in a python process related to the GIL
# it is likely an exit handler is missing.
AnalysisDataService = lazy_instance_access(AnalysisDataServiceImpl,
                                           onexit=AnalysisDataServiceImpl.clear)
AlgorithmFactory = lazy_instance_access(AlgorithmFactoryImpl)
AlgorithmManager = lazy_instance_access(AlgorithmManagerImpl,
                                        onexit=AlgorithmManagerImpl.clear)
FileFinder = lazy_instance_access(FileFinderImpl)
FileLoaderRegistry = lazy_instance_access(FileLoaderRegistryImpl)
FrameworkManager = lazy_instance_access(FrameworkManagerImpl,
                                        onexit=FrameworkManagerImpl.shutdown)
FunctionFactory = lazy_instance_access(FunctionFactoryImpl)
WorkspaceFactory = lazy_instance_access(WorkspaceFactoryImpl)
CatalogManager = lazy_instance_access(CatalogManagerImpl)

# backwards-compatible
mtd = AnalysisDataService
