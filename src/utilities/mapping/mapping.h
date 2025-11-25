#pragma once
#include "../functions.h"

namespace utils
{
////////////////////////////////////////////////////////////
// Map functions to threads.

template< class int_T = int >
__forceinline__ __device__ int_T th_id(  )
{
    return int_T( threadIdx.x ) + blockDim.x * blockIdx.x;
}

template< typename fun_T >
__forceinline__ __device__ void map_1d
( const fun_T & f, const int & length,
  const int   & offset = 0 )
{
    for( int i = threadIdx.x; i < length; i += blockDim.x )
        f( i + offset );
    return;
}

////////////////////////////////////////////////////////////
// Multidimensional mappings

template< class idx_T > __forceinline__ __device__ __host__
void muldim( idx_T & i, const int & l_d, const int & n_dim )
{
    i[ 0 ] = l_d & 1 ;
    for( int a = 1 ; a < n_dim; ++ a )
         i [ a ] = ( l_d >> a ) &  1 ;
    return;
}

template  < class I , class K > __forceinline__ __device__
void didx_refine( I & di, const int & l, const K & ax )
{
    di[ ax[ 0 ] ] =              0;
    di[ ax[ 1 ] ] =   l        & 1;
    di[ ax[ 2 ] ] = ( l >> 1 ) & 1;        
    return;
}

__forceinline__ __device__ __host__ void
divmod( int & p, int & r, const int & n, const int & m )
{
    p =     n / m;
    r = n - p * m;
    return;
}

__forceinline__ __device__ __host__
int mod3( const int & i )
{
    return i < 3 ? i : i - 3;
}

template< typename f_T > __forceinline__ __device__
void idx_swap( f_T * x, const int & axis )
{
    if( mod3( axis ) != 0 )
    {
        f_T s[ 3 ];
        for( int i = 0; i < 3; ++ i )
            s[ i ] = x[ utils::mod3( i + axis ) ];
        for( int i = 0; i < 3; ++ i )
            x[ i ] = s[ i ];
    }
    return;
};
};                              // namespace utils
