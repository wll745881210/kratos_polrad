#pragma once
#include <type_traits>

namespace crtp
{
////////////////////////////////////////////////////////////
// CRTP-related types and helpers

struct dummy_t {  };

template< template< typename > class  this_T, typename d_T,
          template< typename > class super_T >
using helper_t = super_T
< std::conditional_t< std::is_same< dummy_t  , d_T >::value,
                            this_T< dummy_t >, d_T > >;

template< class ... Ts >
struct last
{
    template < class T > struct tag { using type = T; };
    using type
    = typename decltype( ( tag< Ts >{  }, ... ) )::type;
}; 

template< template < class... > class this_T,  class... T >
using  helper = std::conditional_t
< std::is_same_v< typename last< T... >::type, dummy_t >,
  this_T< T... >, typename last< T... >::type >;

#define __crtp_def_self__( THIS_T, DERIVED_T, ... )  using \
this_t = THIS_T< __VA_ARGS__ __VA_OPT__(,) DERIVED_T >;    \
template< typename U = DERIVED_T > __device__ __host__     \
__forceinline__ std::enable_if_t<   std::is_same           \
    < U, crtp::dummy_t >::value, const    this_t & >       \
get_self(  ) const                                         \
{                                                          \
    return ( * this );                                     \
};                                                         \
template< typename U = DERIVED_T > __device__ __host__     \
__forceinline__ std::enable_if_t< ! std::is_same           \
    < U, crtp::dummy_t >::value, const DERIVED_T & >       \
get_self(  ) const                                         \
{                                                          \
    return static_cast< const DERIVED_T & > ( * this );    \
};                                                         \
template< typename U = DERIVED_T > __device__ __host__     \
__forceinline__ std::enable_if_t<   std::is_same           \
    < U, crtp::dummy_t >::value,    this_t & >             \
get_self_c(  ) const                                       \
{                                                          \
    return const_cast< this_t & > ( * this );              \
};                                                         \
template< typename U = DERIVED_T > __device__ __host__     \
__forceinline__ std::enable_if_t< ! std::is_same           \
    < U, crtp::dummy_t >::value,  DERIVED_T & >            \
get_self_c(  ) const                                       \
{                                                          \
    return const_cast<       DERIVED_T & >                 \
        ( static_cast< const DERIVED_T & > ( * this ) );   \
};                                                        
};                              // namespace crtp
