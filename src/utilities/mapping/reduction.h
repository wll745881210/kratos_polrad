#pragma once

namespace utils
{
////////////////////////////////////////////////////////////
// Atomic reduction; pass the operation via template arg.

__device__ __forceinline__
float atomic_min( float * des, const float x )
{
    int * d_i  = ( int * )  des;
    int   old  = ( * d_i );
    int   assumed( 0 );
    do
    {
        assumed = old;
        old   = atomicCAS
              ( d_i, assumed, __float_as_int
              ( __min1( x, __int_as_float( assumed ) ) ) );
    }   while( assumed != old );
    return __int_as_float( old );
}

__device__ __forceinline__
double atomic_min( double * des, const double x )
{
    typedef unsigned long long int __ul;
    __ul * d_i( ( __ul * ) des );
    __ul   old( * d_i );
    __ul   a  (   0   );
    do
    {
        a = old;
        const auto & b = __min2
                          ( x,__longlong_as_double( a ) );
        const auto & c =__ul( __double_as_longlong( b ) );
        old = atomicCAS( d_i, a, c );
    }   while( a != old );
    return __longlong_as_double( old );
}

__device__ __forceinline__
float atomic_max( float * des, const float x )
{
    int * d_i  = ( int * )  des;
    int   old  = ( * d_i );
    int   assumed( 0 );
    do
    {
        assumed = old;
        old = atomicCAS
          ( d_i, assumed, __float_as_int
          ( utils::max( x, __int_as_float( assumed ) ) ) );
    }   while( assumed != old );
    return __int_as_float( old );
}

__device__ __forceinline__
double atomic_max( double * des, const double x )
{
    typedef unsigned long long int __ul;
    __ul * d_i( ( __ul * ) des );
    __ul   old( * d_i );
    __ul   a  (   0   );
    do
    {
        a = old;
        const auto & b = utils::max
                          ( x,__longlong_as_double( a ) );
        const auto & c =__ul( __double_as_longlong( b ) );
        old = atomicCAS( d_i, a, c );
    }   while( a != old );
    return __longlong_as_double( old );
}

template< class f_T >
__device__ __forceinline__
void atomic_set( f_T * des, const f_T & x )
{
    if constexpr( sizeof( f_T ) == sizeof( int ) )
    {
        auto * p( ( int * )( des ) );
        const auto tgt = __float_as_int( x );
        int old( * p ), assumed;
        do
        {
            assumed = old;
            old = atomicCAS( p, assumed, tgt );
        }   while( old != tgt );
    }
    else
    {
        auto * p( ( unsigned long long int * )( des ) );
        const auto tgt = __double_as_longlong( x );
        unsigned long long old( * p ), assumed;
        do
        {
            assumed = old;
            old = atomicCAS( p, assumed, tgt );
        }   while( old != assumed );            
    }
    return;
}

template< class int_T >
__device__ __forceinline__ int_T atomic_inc( int_T * p )
{
    int_T res( 0 );
    if constexpr( std::is_same_v< int_T, size_t > )
        res = atomicAdd( ( unsigned long long * ) p,
                         ( unsigned long long ) 1 );
    else
        res = atomicAdd( p, ( int_T ) 1 );
    return res;
}

////////////////////////////////////////////////////////////
// Block-wise reduction operation

template< typename F, F ( * f )( F, F ) >
__device__ void block_reduce( F * x, int n )
{
    __syncthreads(  );
    const auto & i = threadIdx.x;
    while( n > 1 )
    {
        const int & dn = ( n + 1 )   >> 1;
        const int & di =  dn + i ;
        if( di < n )
            x[ i ] = f( x[ i ], x[ di ] );
        __syncthreads(  );
        n = dn;
    }
    return;
}

template< class V, class fun_T > __forceinline__ __device__
void block_reduce( V & x, const fun_T & f, int n )
{
    __syncthreads(  );
    const auto & i = threadIdx.x;
    while( n > 1 )
    {
        const int & dn = ( n + 1 )   >> 1;
        const int & di =  dn + i ;
        if( di < n )
            x[ i ] = f( x[ i ], x[ di ] );
        __syncthreads(  );
        n = dn;
    }
    return;
}

template< class F >
__device__ __forceinline__ F sum( F a, F b )
{
    return a + b;
};

};                              // namespace utils
