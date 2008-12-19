#ifndef MANTID_ALGORITHM_CHANGEBINOFFSET_H_
#define MANTID_ALGORITHM_CHANGEBINOFFSET_H_

//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------
#include "MantidAPI/Algorithm.h"
#include "MantidAPI/Workspace.h"

namespace Mantid
{
  namespace Algorithms
  {
    /**Takes a workspace and adjusts all the time bin values by the same amount.

    Required Properties:
    <UL>
    <LI> InputWorkspace - The name of the Workspace to take as input </LI>
    <LI> OutputWorkspace - The name of the workspace in which to store the result </LI>
    <LI> Offset - The number by which to change the time bins by</LI>
    </UL>
	  
    @author 
    @date 11/07/2008

    Copyright &copy; 2008 STFC Rutherford Appleton Laboratories

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

    File change history is stored at: <https://svn.mantidproject.org/mantid/trunk/Code/Mantid>    
    Code Documentation is available at: <http://doxygen.mantidproject.org>
    */
    class DLLExport ChangeBinOffset : public API::Algorithm
    {
    public:
      /// Default constructor
      ChangeBinOffset() : API::Algorithm() {};
      /// Destructor
      virtual ~ChangeBinOffset() {};
      /// Algorithm's name for identification overriding a virtual method
      virtual const std::string name() const { return "ChangeBinOffset";}
      /// Algorithm's version for identification overriding a virtual method
      virtual const int version() const { return 1;}
      /// Algorithm's category for identification overriding a virtual method
      virtual const std::string category() const { return "General";}

    private:
      // Overridden Algorithm methods
      void init();
      void exec();
    
       API::Workspace_sptr createOutputWS(API::Workspace_sptr input);
            
      /// Static reference to the logger class
      static Mantid::Kernel::Logger& g_log;
    };

  } // namespace Algorithm
} // namespace Mantid

#endif /*MANTID_ALGORITHM_CHANGEBINOFFSET_H_*/
