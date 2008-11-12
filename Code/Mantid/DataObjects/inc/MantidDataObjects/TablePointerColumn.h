#ifndef MANTID_DATAOBJECTS_TABLEPOINTERCOLUMN_H_
#define MANTID_DATAOBJECTS_TABLEPOINTERCOLUMN_H_

//----------------------------------------------------------------------
// Includes
//----------------------------------------------------------------------

#include "MantidDataObjects/Column.h"
#include "MantidKernel/Logger.h"

#include <iostream>
#include <vector>
#include <typeinfo>
#include <stdexcept>
#include <boost/shared_ptr.hpp>

namespace Mantid
{

//----------------------------------------------------------------------
// Forward declarations
//----------------------------------------------------------------------

namespace DataObjects
{

/** \class TablePointerColumn

    


    \author Roman Tolchenov
    \date 7/11/2008

    Copyright &copy; 2007-8 STFC Rutherford Appleton Laboratory

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

    File change history is stored at: <https://svn.mantidproject.org/mantid/trunk/Code/Mantid>.
    Code Documentation is available at: <http://doxygen.mantidproject.org>
*/
template <class Type>
class DLLExport TablePointerColumn: public Column
{
public:

    /// Virtual destructor.
    virtual ~TablePointerColumn(){}
    /// Number of individual elements in the column.
    int size(){return int(m_data.size());}
    /// Reference to the data.
    Type& data(int i){return *m_data[i];}
    const std::type_info& get_type_info(){return typeid(Type);}
    const std::type_info& get_pointer_type_info(){return typeid(Type*);}
protected:
    /// Resize.
    void resize(int count)
    {
        if (count > int(m_data.size()))
        {
            while(count > int(m_data.size()))
            {
                m_data.push_back(boost::shared_ptr<Type>(new Type));
            }
        }
        else if (count < int(m_data.size()))
        {
            m_data.erase(m_data.begin()+count,m_data.end());
        }
    }
    /// Inserts default value at position index. 
    void insert(int index){m_data.insert(m_data.begin()+index,boost::shared_ptr<Type>(new Type));}
    /// Removes an item.
    void remove(int index){m_data.erase(m_data.begin()+index);}
    void* void_pointer(int index){return &(*m_data[index]);}
private:
    std::vector< boost::shared_ptr<Type> > m_data;
    friend class TableWorkspace;
};

template<class T>
class TablePointerColumn_ptr: public boost::shared_ptr<TablePointerColumn<T> >
{
public:
    TablePointerColumn_ptr(boost::shared_ptr<Column> c):boost::shared_ptr<TablePointerColumn<T> >(boost::dynamic_pointer_cast<TablePointerColumn<T> >(c))
    {
        if (this->get() == NULL)
        {
            Kernel::Logger& log = Kernel::Logger::get("TableWorkspace");
            std::string str = "Data type of column "+c->name()+" does not match "+typeid(T).name();
            log.error(str);
            throw std::runtime_error(str);
        }
    }
};

/*template <class Type>
TableVector<Type>& TableColumn<Type>::data()
{
    return static_cast< TableVector<Type>& >(m_data);
}*/

} // namespace DataObjects
} // Namespace Mantid

#define DECLARE_TABLEPOINTERCOLUMN(DataType,TypeName) \
    namespace{ \
    Mantid::Kernel::RegistrationHelper register_pointer_column_##TypeName(  \
    (Mantid::DataObjects::ColumnFactory::Instance().subscribe< Mantid::DataObjects::TablePointerColumn< DataType > >(#TypeName),0)); \
    } 

#endif /*MANTID_DATAOBJECTS_TABLEPOINTERCOLUMN_H_*/
