#include "MantidDataObjects/Workspace1D.h"
#include "MantidAPI/tripleRef.h"
#include "MantidAPI/TripleIterator.h"
#include "MantidAPI/tripleIteratorCode.h"

DECLARE_WORKSPACE(Workspace1D)

namespace Mantid
{
namespace DataObjects
{

/// Constructor
Workspace1D::Workspace1D() : API::Workspace(), 
			     Histogram1D()
{ }

  /// Copy Constructor
Workspace1D::Workspace1D(const Workspace1D& A) :
  API::Workspace(A),Histogram1D(A)
{ }

/*!
    Assignment operator
    \param A :: Workspace  to copy
    \return *this
   */
Workspace1D& 
Workspace1D::operator=(const Workspace1D& A)
{
  if (this!=&A)
    {
      API::Workspace::operator=(A);
      Histogram1D::operator=(A);
    }
  return *this;
}

/// Destructor
Workspace1D::~Workspace1D()
{}


template DLLExport class Mantid::API::triple_iterator<DataObjects::Workspace1D>;

} // namespace DataObjects
} //NamespaceMantid
