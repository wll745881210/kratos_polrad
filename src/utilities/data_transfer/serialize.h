#pragma once

#include <iostream>
#include <sstream>
#include <type_traits>
#include <utility>
#include <array>
#include <vector>
#include <map>
#include <set>
#include <list>

struct serialize
{
////////////////////////////////////////////////////////////
// Interface type

struct interface
{
    virtual std::ostream & write
    ( std::ostream & os )  const = 0;
    virtual std::istream &  read
    ( std::istream & is )        = 0;
};

////////////////////////////////////////////////////////////
// Type-traits

template < typename    args_T >
struct is_serialized                 : std::false_type {  };
template < typename... args_T >
struct is_serialized
< std::basic_string  < args_T... > > : std:: true_type {  };
template < typename... args_T >
struct is_serialized
< std::vector        < args_T... > > : std:: true_type {  };
template < typename... args_T >
struct is_serialized
< std::array         < args_T... > > : std:: true_type {  };
template < typename... args_T >
struct is_serialized
< std::map           < args_T... > > : std:: true_type {  };
template < typename... args_T >
struct is_serialized
< std::set           < args_T... > > : std:: true_type {  };
template < typename... args_T >
struct is_serialized
< std::list          < args_T... > > : std:: true_type {  };
template < typename... args_T >
struct is_serialized
< std::pair          < args_T... > > : std:: true_type {  };

template< typename T, bool positive = true >
using if_serialized_t = std::enable_if_t
< ! ( positive ^ ( std::is_base_of< interface, T >::value ||
                   is_serialized< T >::value ) ) ,    int >;

////////////////////////////////////////////////////////////
// Interface read/write

template< typename T, std::enable_if_t< std::is_base_of
                      < interface, T >::value, int > = 0 >
static std::istream &  read
( std::istream & is,       T & t )
{
    return t.read ( is );
}

template< typename T, std::enable_if_t< std::is_base_of
                      < interface, T >::value, int > = 0 >
static std::ostream & write
( std::ostream & os, const T & t )
{
    return t.write( os );
}

////////////////////////////////////////////////////////////
// Basic datatypes

template< typename T, if_serialized_t< T, false > = 0 >
static std::istream & read
( std::istream & is,       T & t )
{
    is.read ( ( char * ) & t, sizeof( T ) );
    return is;
}

template< typename T, if_serialized_t< T, false > = 0 >
static std::ostream & write
( std::ostream & os, const T & t )
{
    os.write( ( char * ) & t, sizeof( T ) );
    return os;
}

////////////////////////////////////////////////////////////
// char *

static std::istream & read
( std::istream & is,       char * str, const size_t & size )
{
    is.read ( str, size );
    return is;
}

static std::ostream & write
( std::ostream & os, const char * str, const size_t & size )
{
    os.write( str, size );
    return os;
}

////////////////////////////////////////////////////////////
// Plain array

template< typename T, if_serialized_t< T, false > = 0 >
static std::istream & read
( std::istream & is,       T a[  ], size_t size )
{
    read ( is, ( char * ) a, size * sizeof( T ) );
    return is;
}

template< typename T, if_serialized_t< T,  true > = 0 >
static std::istream & read
( std::istream & is,       T a[  ], size_t size )
{
    for( size_t i = 0; i < size; ++ i )
        read ( is, a[ i ] );
    return is;
}

template< typename T, if_serialized_t< T, false > = 0 >
static std::ostream & write
( std::ostream & os, const T a[  ], size_t size )
{
    write( os, ( const char * ) a, size * sizeof( T ) );
    return os;
}

template< typename T, if_serialized_t< T,  true > = 0 >
static std::ostream & write
( std::ostream & os, const T a[  ], size_t size )
{
    for( size_t i = 0; i < size; ++ i )
        write( os, a[ i ] );
    return os;
}

////////////////////////////////////////////////////////////
// Generic array; read's have callbacks

template< typename V, typename F, if_serialized_t
        < typename V::value_type, false > = 0 >
static std::istream &  read_arr
( std::istream & is, V & v, F & f )
{
    size_t size( 0 );
    read( is, size );
    f       ( size );
    read ( is, ( char * ) & v.front(  ),
           size * sizeof( typename V::value_type ) );
    return is;
}

template< typename V, typename F, if_serialized_t
        < typename V::value_type,  true > = 0 >
static std::istream &  read_arr
( std::istream & is, V & v, F & f )
{
    size_t size( 0 );
    read( is, size );
    f       ( size );
    for( auto & v_s : v )
        read (  is, v_s );
    return is;
}

template< typename V, if_serialized_t
        < typename V::value_type, false > = 0 >
static std::ostream & write_arr
( std::ostream & os, const V & v )
{
    const size_t size = v.size(  );
    write( os, size );
    write( os, ( const char * ) & v.front(  ),
           size * sizeof( typename V::value_type ) );
    return os;
}

template< typename V, if_serialized_t
        < typename V::value_type,  true > = 0 >
static std::ostream & write_arr
( std::ostream & os, const V & v )
{
    const size_t size = v.size(  );
    write( os, size );
    for( const auto & v_s : v )
        write( os,    v_s );
    return os;
}

////////////////////////////////////////////////////////////
// std::pair

template< typename T1, typename T2 >
static std::istream & read
( std::istream & is,       std::pair< T1, T2 > & p )
{
    read ( is, p.first  );
    read ( is, p.second );
    return is;
}

template< typename T1, typename T2 >
static std::ostream & write
( std::ostream & os, const std::pair< T1, T2 > & p )
{
    write( os, p.first  );
    write( os, p.second );
    return os;
}

////////////////////////////////////////////////////////////
// std::string

static std::istream & read
( std::istream & is,       std::string & str )
{
    size_t size(   0 );
    read( is,   size );
    str.resize( size );
    is.read ( & str[ 0 ], size );    
    return is;
}

static std::ostream & write
( std::ostream & os, const std::string & str )
{
    const size_t size = str.size(  );
    write   ( os,             size );
    os.write( str.c_str(  ),  size );
    return os;
}

////////////////////////////////////////////////////////////
// std::array

template< typename T, size_t N >
static std::istream & read
( std::istream & is,       std::array< T, N > & a )
{
    return read_arr( is, a, [ & ] ( const size_t & size )
    {
        if( size != a.size(  ) )
            throw std::runtime_error( "Inconsistent size" );
    } );
}

template< typename T, size_t N >
static std::ostream & write
( std::ostream & os, const std::array< T, N > & a )
{
    return write_arr( os, a );
}

////////////////////////////////////////////////////////////
// std::vector

template< typename T >
static std::istream & read
( std::istream & is,       std::vector< T > & v )
{
    return read_arr( is, v, [ & ] ( const size_t & size )
                            { v.resize( size ); } );
}

template< typename T >
static std::ostream & write
( std::ostream & os, const std::vector< T > & v )
{
    return write_arr( os, v );
}

////////////////////////////////////////////////////////////
// std::map

template< typename K, typename V, typename L >
static std::istream & read
( std::istream & is,       std::map< K, V, L > & m )
{
    size_t size( 0 );
    read( is, size );
    K key;
    m.clear(  );
    for( size_t i = 0; i < size; ++ i )
    {
        read( is,    key   );
        read( is, m[ key ] );
    }
    return is;
}

template< typename K, typename V, typename L >
static std::ostream & write
( std::ostream & os, const std::map< K, V, L > & m )
{
    const size_t size = m.size(  );
    write( os, size );
    for( const auto & p_s : m )
        write( os,    p_s );
    return os;
}

////////////////////////////////////////////////////////////
// std::set

template< typename T >
static std::istream & read
( std::istream & is,       std::set< T > & s )
{
    size_t size( 0 );
    read( is, size );
    s.clear(  );
    T t;
    for( size_t i = 0; i < size; ++ i )
    {
        read( is, t );
        s.insert( t );
    }
    return is;
}

template< typename T >
static std::ostream & write
( std::ostream & os, const std::set< T > & s )
{
    const size_t size = s.size(  );
    write( os, size );
    for( const auto & s_s : s )
        write( os,    s_s );
    return os;
}

////////////////////////////////////////////////////////////
// std::list

template< typename T >
static std::istream & read
( std::istream & is,       std::list< T > & l )
{
    size_t size( 0 );
    read( is, size );
    l.clear(  );
    for( size_t i = 0; i < size; ++ i )
    {
        l  .emplace_back(  );
        read( is, l.back(  ) );
    }
    return is;
}

template< typename T >
static std::ostream & write
( std::ostream & os, const std::list< T > & l )
{
    const size_t size = l.size(  );
    write( os, size );
    for( const auto & l_s : l )
        write( os,    l_s );
    return os;
}

////////////////////////////////////////////////////////////
// If the user just wants or has a string

template< typename T >
static std::string get( const T & t )
{
    std::stringstream ss;
    write( ss, t );
    return ss.str(  );
}

template< typename T >
static void set
( T & t, const char * c, const size_t & size )
{
    std::stringstream ss;
    ss.write( c, size );
    ss.clear(         );
    read    ( ss,   t );
    return;
}

};                              // struct serialize
