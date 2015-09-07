#ifndef MANTID_CUSTOMINTERFACES_REFLGENERATENOTEBOOK_H
#define MANTID_CUSTOMINTERFACES_REFLGENERATENOTEBOOK_H

/** @class ReflGenerateNotebook

    This class creates ipython notebooks from the ISIS Reflectometry
    (Polref) interface

    Copyright &copy; 2007-8 ISIS Rutherford Appleton Laboratory, NScD Oak Ridge
    National Laboratory & European Spallation Source

    This file is part of Mantid.

    Mantid is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 3 of the License, or
    (at your option) any later version.

    Mantid is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    File change history is stored at: <https://github.com/mantidproject/mantid>.
    Code Documentation is available at: <http://doxygen.mantidproject.org>
    */

#include "MantidQtCustomInterfaces/QReflTableModel.h"
#include "MantidKernel/System.h"

#include <set>
#include <map>
#include <string>
//#include <tuple>

namespace MantidQt {
  namespace CustomInterfaces {

    class DLLExport ReflGenerateNotebook {
    public:

      ReflGenerateNotebook(std::string name,
                           QReflTableModel_sptr model,
                           const std::string instrument,
                           const int COL_RUNS, const int COL_TRANSMISSION, const int COL_OPTIONS, const int COL_ANGLE,
                           const int COL_QMIN, const int COL_QMAX, const int COL_DQQ, const int COL_SCALE);

      virtual ~ReflGenerateNotebook(){};

      void generateNotebook(std::map<int, std::set<int>> groups, std::set<int> rows, std::string filename);

    private:

      std::string plotIvsQ(std::vector<std::string> ws_names);

      std::tuple<std::string, std::string> stitchGroupString(std::set<int> rows);

      std::tuple<std::string, std::string> reduceRowString(int rowNo);

      std::tuple<std::string, std::string> loadWorkspaceString(std::string runStr);

      std::string plusString(std::string input_name, std::string output_name);

      std::tuple<std::string, std::string> loadRunString(std::string run);

      std::string getRunNumber(std::string ws_name);

      std::tuple<std::string, std::string> scaleString(std::string runNo, double scale);

      std::tuple<std::string, std::string> convertToPointString(std::string wsName);

      template<typename T, typename A>
      std::string vectorParamString(std::string param_name, std::vector<T,A> &param_vec);

      std::tuple<std::string, std::string> rebinString(int rowNo, std::string runNo);

      std::tuple<std::string, std::string> transWSString(std::string trans_ws_str);

      std::string m_wsName;
      QReflTableModel_sptr m_model;
      const std::string m_instrument;

      const int COL_RUNS;
      const int COL_TRANSMISSION;
      const int COL_OPTIONS;
      const int COL_ANGLE;
      const int COL_QMIN;
      const int COL_QMAX;
      const int COL_DQQ;
      const int COL_SCALE;

    };

  }
}

#endif //MANTID_CUSTOMINTERFACES_REFLGENERATENOTEBOOK_H
