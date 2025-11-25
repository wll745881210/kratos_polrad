#pragma once

#include <functional>
#include <string>
#include <typeinfo>
#include <type_traits>

#include "../../utilities/data_transfer/serialize.h"

// #define __MPI__

namespace binary_io
{
////////////////////////////////////////////////////////////
// Buffered read or not

enum buffer_read_t{ unbuffered, buffered };

////////////////////////////////////////////////////////////
// Buffer auxilliaries

struct buffer_t
{
    size_t size;
    void * data;
    bool   copy;

    buffer_t( const size_t & size = 0 ) : data( nullptr ),
                                          copy(    true )
    {
        this->size  =    0  ;
        this->resize( size );
        return;
    };
    ~buffer_t(  )
    {
        if( copy )
            free( this->data );
        return;
    };
    inline void resize( const  size_t & size )
    {
        if( size != 0 && this->size  != size )
        {
            if( copy )
            {
                free         ( this->data );
                this->data = malloc( size );
            }
            this->size = size;
        }
        if( size == 0 )
            this->data = nullptr;
        return;
    }
};

struct b_info_t : public serialize::interface
{
    size_t   size;
    char   u_size;              // Unit size
    size_t offset;

    std::ostream & write( std::ostream & stream ) const
    {
        serialize::write( stream,          size );
        serialize::write( stream,        u_size );
        serialize::write( stream,        offset );
        return stream;
    };
    std::istream & read ( std::istream & stream )
    {
        serialize::read ( stream,          size );
        serialize::read ( stream,        u_size );
        serialize::read ( stream,        offset );
        return stream;
    };
};

////////////////////////////////////////////////////////////
// Functor types for callbacks before writing/after reading

typedef std::function
< void( b_info_t & , buffer_t & ) > f_callback_t;

////////////////////////////////////////////////////////////
// Type traits

template< typename T, bool p >
using if_serialized_t = serialize::if_serialized_t< T, p >;

template< typename T, bool p >
using if_void_t       = std::enable_if_t
< ! ( p ^ std::is_void< T >::value ), int >;

};                              // namespace binary_io
