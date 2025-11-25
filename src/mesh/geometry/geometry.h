#pragma once

#include <functional>

#include "../../types.h"
#include "../../device/device.h"
#include "../types/dat_3d.h"
#include "../types/functors.h"
#include "region.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Types

using type::   idx_t;
using type:: float_t;
using type::float2_t;

////////////////////////////////////////////////////////////
// Mesh geometry

struct geo_t
{
    ////////// Data //////////
    idx_t               n_ceff;
    int                  n_dim;
    int                   n_gh;
    bool phybnd_flag[ 3 ][ 2 ];
    bool refine_flag[ 3 ][ 2 ];
    bool coarse_flag[ 3 ][ 2 ];

    bool  is_uniform[ 3 ];
    float_t *     xv[ 3 ]; // Cell-center loc, size: n_cell
    float_t *     xf[ 3 ]; // Face-center loc, size: n_face

    float_t              xf0[ 3 ];
    float_t              dx0[ 3 ];
    float_t             sfc0[ 3 ];
    float_t             vol0;
    // dat_3d_t< float_t >  dxf[ 3 ];
    dat_3d_t< float_t >  sfc[ 3 ];
    dat_3d_t< float_t >  vol;

    ////////// Functions //////////
    __host__ geo_t    (  ) ;
    __host__ void setup  ( const idx_t     & n_ceff ,
      const  int  & n_gh , const f_new_t   & f_new  );
    __host__ void setup  ( const geo_t     &   src  ,
                           const f_new_t   & f_new  );    
    __host__ void free   ( const f_free_t  & f_free );
    __host__ void cp_prim( geo_t           & tgt    ) const;
    __host__ void cp_shlw( geo_t           & tgt    ) const;
    __host__ void cp_deep( geo_t           & tgt    ,
                           const f_cp_t    & f_cp   ) const;
    __host__ void read   ( const f_new_t   & f_new  ,  
                           const f_read_t  & f_r    );
    __host__ void write  ( const f_write_t & f_w    ) const;

    ////////// Coordinate access //////////
    __host__ __device__ __forceinline__ float_t x_cc
    ( const int & ax, const int & i )   const
    {
        return   dx0[ ax ] > 0 
             ? ( xf0[ ax ] + ( i + 0.5f ) * dx0[ ax ] )
             :   xv [ ax ]   [ i + n_gh ] ;
    };
    __host__ __device__ __forceinline__ float_t x_fc
    ( const int & ax, const int & i )   const
    {
#if defined(__CPU_DEBUG__) || defined( __GPU_DEBUG__ )
        if( i + n_gh < 0 ||
            i + n_gh > n_ceff[ ax ] + n_gh * 2 )
#ifdef     __CPU_DEBUG__
                throw std::out_of_range( "x_fc" );
#else
                printf( "xf: %d\n", i );
#endif  // __CPU_DEBUG__            
#endif  // __CPU_DEBUG__ || __GPU_DEBUG__            
        return   dx0[ ax ] > 0
             ? ( xf0[ ax ] + i *  dx0 [ ax ] )
             :   xf [ ax ] [ i + n_gh ] ;        
    };
    template < class idx_T >
    __host__ __device__ __forceinline__ float_t dx_f
    ( const int & a, const idx_T & i )  const
    {
        return dx0 [ a ] > 0 ? dx0[ a ]   :
             // ( dxf [ a ]     ? dxf[ a ] ( i ) :
            ( x_fc( a, i[ a ] +  1 ) - x_fc( a, i[ a ] ) );
    };
    template< class idx_T > __forceinline__  __device__
    __host__ void x_cc( coord_t & x, const idx_T & i ) const
    {
        for( int a = 0; a < 3; ++ a )
            x[ a ] = x_cc ( a, i[ a ] );
        return ;
    };
    template< class idx_T > __forceinline__  __host__
    coord_t x_cc( const idx_T & i ) const
    {
        coord_t res;
        x_cc  ( res, i );
        return  res;
    };    
    template< class idx_T >
    __host__ __device__ __forceinline__ float_t
    surface( const idx_T & i,  const int & axis ) const
    {
        return ( sfc0[ axis ]  > 0 ? sfc0[ axis ] :
                 sfc [ axis ]( i ) );
    };
    template< class idx_T >
    __host__ __device__ __forceinline__ float_t
    volume ( const idx_T & i ) const
    {
        return ( vol0 > 0 ? vol0 : vol( i ) );
    };

    template< class f_T >  __forceinline__ __device__
    __host__ int idx( const f_T & x, const int & ax ) const
    {
        if( is_uniform[ ax ] )
            return ( x - x_fc( ax, 0 ) ) / dx0[ ax ];
        if( x <  x_fc( ax, 0 ) ||
            x >= x_fc( ax, n_ceff[ ax ] ) )
            return -1;
        for( int i = 0; i < n_ceff[ ax ]; ++ i )
            if( x_fc( ax, 1 + i ) >  x  )
                return i;
        return  -1;
    };
    template< class idx_T, class x_T >
    __forceinline__ __device__ __host__
    void idx( idx_T & i, const x_T & x ) const
    {
        for( int a = 0 ; a < 3; ++ a )
             i [ a ] = ( a < n_dim ? idx( x[ a ], a ) : 0 );
        return;
    };
};                              // struct   geo_t
};                              // namespace mesh
