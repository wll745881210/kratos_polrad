#pragma once

#include "../../types.h"

#include <deque>
#include <list>
#include <vector>
#include <array>
#include <type_traits>

namespace type
{
////////////////////////////////////////////////////////////
// Sequential containers
template < typename  T >
struct is_seq              : std::false_type {  };
template < typename... Args >
struct is_seq
< std::vector< Args... > > : std:: true_type {  };
template < typename... Args >
struct is_seq
< std::deque < Args... > > : std:: true_type {  };
template < typename... Args >
struct is_seq
< std::list  < Args... > > : std:: true_type {  };

////////////////////////////////////////////////////////////
// std::array
template < typename T >
struct is_std_array        : std::false_type {  };
template < typename T, std::size_t N >
struct is_std_array
< std::array< T, N > >     : std:: true_type {  };

////////////////////////////////////////////////////////////
// v3_base_t
template < typename T >
struct is_vec_3d           : std::false_type {  };
template < typename T >
struct is_vec_3d
< type::v3_base_t < T > >  : std:: true_type {  };

////////////////////////////////////////////////////////////
// Type enable_if helpers
template< typename T, template< typename >
          class tester_T, bool  positive = true >
using if_t = std::enable_if_t
< ! ( positive  ^ tester_T< T >::value ) ,  int >;

////////////////////////////////////////////////////////////
// Get the "real" type, not the pointer

template< typename T >
using prim_type_t = typename std::remove_pointer_t
    < std::remove_reference_t< std::remove_cv_t< T > > >;

template< typename T > __forceinline__
constexpr size_t size( const T & t )
{
    return sizeof( std::remove_cv_t
                 < std::remove_pointer_t< T > > );
};

////////////////////////////////////////////////////////////
// Fences for compile-time checks
template  < typename T =  void >
constexpr auto  always_false_v = false;    

};                              // namespace type
