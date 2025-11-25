#pragma once

namespace mesh
{
////////////////////////////////////////////////////////////
// 1D data access interface

template< class f_T >
struct dat_1d_t
{
    f_T * dat;                  // Data
    f_T * da0;                  // Data w/o ghost zones
    int n_int;                  // Internal dimension
    int n_spa;                  // Spatial  dimension
    int n_gh ;

    __device__ __forceinline__
    f_T * init( void      *  p_sh, const int & n_spa ,
                const int & n_int, const int &  n_gh )
    {
        this->n_gh  = n_gh ;
        this->n_int = n_int;
        this->n_spa = n_spa + 2 * n_gh;
        da0  = ( f_T * ) ( p_sh );
        dat  = da0 + n_int * n_gh;
        return da0 + n_int * this->n_spa;
    };
    __device__ __forceinline__ void * init
    ( void * p_sh, const int & n_spa, const int & n_gh )
    {
        return init( p_sh, n_spa, 1, n_gh );
    };
    __device__ __forceinline__ f_T * at
    ( const int & i_spa ) const
    {
#ifdef __CPU_DEBUG__
        if( i_spa < -n_gh || i_spa >= n_spa - n_gh )
            throw std::out_of_range ( "dat_1d" );
#endif
#ifdef __GPU_DEBUG__
        if( i_spa < -n_gh || i_spa >= n_spa - n_gh )
            printf( "1d" );
#endif
        return const_cast< f_T * > ( dat ) + i_spa * n_int;
    };
    __device__ __forceinline__ f_T & operator[  ]
    ( const int &   idx ) const
    {
        return const_cast< f_T * > ( da0 ) [ idx ];
    };
    __device__ __forceinline__ f_T & operator(  )
    ( const int &   idx ) const
    {
        return const_cast< f_T * > ( dat ) [ idx ];
    };
    __device__ __forceinline__ f_T & operator(  )
    ( const int & i_spa,  const int & i_int ) const
    {
        return  this->at  ( i_spa ) [ i_int ];
    };
};

};                              // namespace mesh
