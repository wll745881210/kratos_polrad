#pragma once

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "binary_io_types.h"

namespace binary_io
{
////////////////////////////////////////////////////////////
// class binary_io::base_t                                //
// File structure: see "binary_io.h"                      //

class base_t
{
    ////////// Con-/destructor //////////
public:                         // Functions
     base_t(  );
    ~base_t(  );
    virtual void finalize(  ) {  };

    ////////// Endianness //////////
protected:                      // bool: is little endian?
    bool                                 is_le;

    ////////// Data //////////
protected:                      // Data
    buffer_read_t                  buffer_read;
    size_t                         header_size;
    size_t                         proc_offset;
protected:                      // Data
    std::map< std::string,  buffer_t >  buffer;
public:
    std::map< std::string,  b_info_t >  b_info;

    ////////// File //////////
protected:                      // Data
    std::fstream                  fio;
    std::ios_base::openmode file_mode;
public:                         // Data
    std::string             file_name;
public:                         // Functions
    virtual void open
    ( const std::string   & name, const std::string & mode,
      const buffer_read_t & buffer_read = unbuffered );
    virtual void close(  );

    ////////// Save and load (overall) //////////
protected:                      // Type
    typedef std::fstream::openmode o_t;
protected:                      // Function
    virtual void mpi_warning(  );
    virtual void prep_header(  );
    virtual void save_header(  );
    virtual void save_bulk  (  );
    virtual void load_bulk  (  );
    virtual void load_header(  );
    virtual void check_open
    ( const o_t & desired_mode );
public:                         // Functions
    virtual void save       (  );
    virtual void load       (  );
    virtual void clear( const bool & deep = false );

    ////////// Information //////////
public:                         // Functions
    const b_info_t & info( const std::string & tag );
    template< typename T >
    size_t      get_count( const std::string & tag );

    ////////// Read and write //////////
public:                         // Generic
    virtual void  read_callback
    ( const std::string & tag, const f_callback_t & f );
    virtual void write_callback
    ( const std::string & tag, const f_callback_t & f );
    virtual void   f_read_at
    (       void   * data, const size_t & size,
      const size_t & offset );
    virtual void   f_read_at_all
    (       void   * data, const size_t & size,
      const size_t & offset );
    virtual size_t f_write_at
    ( const void   * data, const size_t & size,
      const size_t & offset );    
public:                         // Read
    virtual   bool  has_tag ( const std::string & tag );
    size_t    read( void * p, const std::string & tag );
    template < typename T, if_void_t      < T, false > = 0 >
    size_t    read( T    * p, const std::string & tag );
    template< typename T,  if_serialized_t< T, false > = 0 >
    size_t    read( T    & t, const std::string & tag );
    template< typename T,  if_serialized_t< T,  true > = 0 >
    size_t    read( T    & t, const std::string & tag );
public:                         // Write
    virtual size_t write
    ( const void *,  const size_t  &, const size_t &,
      const std::string &, const bool & copy = false );
    template < typename T, if_void_t      < T, false > = 0 >
    size_t write( const T * p, size_t  count,
                  const std::string &    tag,
                  const bool        &   copy = false );
    template < typename T, if_serialized_t< T, false > = 0 >
    size_t write( const T           &      t,
                  const std::string &    tag,
                  const bool        &   copy = false );
    template < typename T, if_serialized_t< T,  true > = 0 >
    size_t write( const T           & t,
                  const std::string & tag );
};

////////////////////////////////////////////////////////////
// Information

template< typename T >
size_t base_t::get_count( const std::string & tag )
{
    const  size_t & size = info( tag ).size;
    if(    size % sizeof( T ) != 0 )
        throw std::runtime_error( "Indivisible size" );
    return size / sizeof( T );
}

////////////////////////////////////////////////////////////
// Read

template < typename T, if_void_t     < T, false > >
size_t base_t::read( T * p, const std::string & tag )
{
    return read( ( void * ) p, tag );
}

template< typename T, if_serialized_t< T, false > >
size_t base_t::read( T & t, const std::string & tag )
{
    const   auto count = read( ( void * )( & t ), tag );
    buffer[ tag ].copy = true; // Guarantee deletion
    return count;
}

template< typename T, if_serialized_t< T,  true > >
size_t base_t::read( T & t, const std::string & tag )
{
    const auto & bi  = b_info.at   ( tag );
    char       * s   = new char[ bi.size ];
    const auto count = read    ( s,  tag );
    serialize::set   ( t, s,     bi.size );
    delete [  ] s;
    return count;
}

////////////////////////////////////////////////////////////
// Write

template < typename T, if_void_t     < T, false > >
size_t base_t::write( const T * p, size_t count,
                      const std::string & tag  ,
                      const bool        & copy )
{
    return write( ( void * ) p, count, sizeof( T ),
                  tag, copy );
}

template< typename T, if_serialized_t< T, false > >
size_t base_t::write( const T & t, const std::string & tag,
                      const bool & copy )
{
    return write( ( void * ) ( & t ), 1, sizeof( T ) , tag,
                  copy );
}

template< typename T, if_serialized_t< T,  true > >
size_t base_t::write( const T & t, const std::string & tag )
{
    const  auto & s = serialize::get( t );
    return write( ( void * )  s.data(   ), s.size(   ), 1,
                  tag, true );
}

};                              // namespace binary_io
