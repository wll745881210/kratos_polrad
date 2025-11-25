#pragma once

#include "../../../mesh/geometry/geometry.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Particle-related types

using sint_t = type::schar_t;
using sidx_t = type::v3_base_t < short >;
enum  todo_t : sint_t{ to_keep = -2, to_rm = -3 };

////////////////////////////////////////////////////////////
// Basic particle type

struct par_base_t
{
    ////////// Geometry //////////
    union
    {
        todo_t    todo;
        sint_t  i_rank;   // This is NOT the rank!
    }   dest;
    type::idx_t             i;
    type::coord_t           x;    
    type::coord_t         dir;
    int                  ib_l;   // Calc: id_l; Comm: id_g
    
    ////////// Radiation Properties //////////
    type::float_t      proper;
    type::  int_t        step;

    ////////// Functions //////////
    __device__ void init( char * p_sh ) {  };
        
    template < class map_T > __forceinline__
    __device__ bool regulate( const map_T & bmap )
    {
        bool res( false );        
        for( int a = 0; a < x.size(  ); ++ a )
            if( bmap.periodic [ a ] )
            {
                if( x[ a ] <  bmap.xlim [ 0 ][ a ] )
                {
                    x[ a ] += bmap.l_box[ a ];
                    res = true;
                }
                else if( x[ a ] >= bmap.xlim [ 1 ][ a ] )
                {   // Must be "else if" in case of low prec
                    x[ a ] -= bmap.l_box[ a ];
                    res = true;
                }
            }
        return res;
    };

    __device__ __host__ __forceinline__
    constexpr size_t size(  )
    {
        return utils::align_c( sizeof( par_base_t ), 8 );
    };
    __host__ __device__ __forceinline__
    void load( const par_base_t & tgt ) 
    {
        ( * this ) = tgt;
        return;
    };
    __host__ __device__ __forceinline__
    void save( par_base_t & tgt ) const
    {
        tgt = ( * this );
        return;
    };
    __host__ __device__ __forceinline__
    void move( par_base_t & tgt ) const
    {
        tgt = ( * this );
        return;
    };        
};

////////////////////////////////////////////////////////////
// "Local" geometry data

struct geo_loc_t
{
    type::coord_t x_l;
    type::coord_t x_r;

    template < class idx_T > __device__ __forceinline__
    void load( const mesh::geo_t & geo, const idx_T & i )
    {
        for( int a = 0; a < x_l.size(  ); ++ a )
        {
            x_l[ a ] = geo.x_fc( a, i[ a ]     );
            x_r[ a ] = geo.x_fc( a, i[ a ] + 1 );
        }
        return;
    };
    template  < class par_T > __device__ __forceinline__
    void shift( const par_T & par, const mesh::geo_t & geo ,
                const int   &  ax, const bool        & rhs )
    {
        if( rhs )
        {
            x_l[ ax ] = x_r[ ax ];
            x_r[ ax ] = geo.x_fc( ax, par.i[ ax ] + 1 );
        }
        else
        {
            x_r[ ax ] = x_l[ ax ];
            x_l[ ax ] = geo.x_fc( ax, par.i[ ax ]     );
        }
        return;
    };
    template< class par_T > __device__ __forceinline__    
    void fix( const mesh::geo_t & geo, const par_T & par )
    {
        for( int a = 0; a < par.x.size(   ) ; ++ a )
            if     ( par.x[ a ] >  x_r[ a ] )
                shift( par, geo, a,  true ) ;
            else if( par.x[ a ] <  x_l[ a ] )
                shift( par, geo, a, false ) ;
        return;
    };

    ////////// Load/save ( not compressed ) //////////
    template < class par_T, class geo_T >
    __forceinline__ __host__ __device__  void  load
    ( par_T & p, geo_loc_t & g_l, const geo_T & geo ) const
    {
        p       =  ( *  this );
        g_l.load( geo, p.idx );
        return;
    };
    template < class par_T >
    __forceinline__ __device__ __host__
    void save( const par_T & p, const geo_loc_t & g_l )
    {
        ( * this ) = p;
        return;
    };
};

};                          // namespace radiation::photon
