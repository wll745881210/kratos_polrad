#pragma once

#include "meshgen.h"
#include "../../utilities/mapping/loop.h"

namespace mesh::meshgen
{
////////////////////////////////////////////////////////////
// Cartesian unifor mesh generator

class cyl_t : public base_t
{
    ////////// Cylindrical geometry //////////
protected:                      // Data
    float2_t ratio;
    float2_t  dr_0;
public:
    virtual void read( const input & args )
    {
        base_t:: read( args );
        dr_0  = -1;
        ratio = args.get< float2_t >
            ( "mesh", "r_ratio", 1 );
        const  auto n_r = n_ceff_global[ 0 ];
        if( fabs( ratio - 1 ) < 1e-6 )
            return;
        if( ratio < 0 )
            ratio = pow( x_lim_global[ 1 ][ 0 ] /
                         x_lim_global[ 0 ][ 0 ], 1. / n_r );
        dr_0  = x_lim_global[ 1 ][ 0 ]
              - x_lim_global[ 0 ][ 0 ];
        dr_0 /= ( pow( ratio, n_r ) - 1 ) / ( ratio - 1 );
        return;
    };
    virtual float2_t location
    ( const float2_t & x_logic, const int & ax ) const
    {
        const  auto & x_lim = x_lim_global;
        if( ax != 0 || fabs( ratio - 1 ) < 1e-6 )
            return x_lim[ 0 ][ ax ] + x_logic *
                 ( x_lim[ 1 ][ ax ] - x_lim [ 0 ][ ax ] );
        const auto n_logic( x_logic * n_ceff_global[ 0 ] );
        return x_lim[ 0 ][ 0 ] + dr_0 / ( ratio - 1 )
             * ( pow( ratio,  n_logic ) - 1 ) ;
    };
    virtual float2_t surface
    ( const idx_t &  i , const mesh::geo_t & geo,
      const int   & ax ) const
    {
        auto  res = base_t::surface( i, geo, ax );
        if( ax == 0 )
            res  *=   geo.x_fc( 0, i[ 0 ] );
        if( ax == 2 )
            res  *= ( geo.x_fc( 0, i[ 0 ] + 1 ) +
                      geo.x_fc( 0, i[ 0 ] ) ) / 2;
        return res;
    };
    virtual float2_t volume
    ( const idx_t & i, const mesh::geo_t & geo ) const
    {
        const auto r_ave = ( geo.x_fc( 0, i[ 0 ] + 1 ) +
                             geo.x_fc( 0, i[ 0 ] ) ) / 2;
        return base_t::volume( i, geo ) * r_ave;
    };
    virtual void set_uniformity ( block_t & b )
    {
        for( int a = 0; a < 3; ++ a )
        {
            b.geo.is_uniform[ a ] = true;
            b.geo.sfc0      [ a ] =   -1;
        }
        b.geo.sfc0[ 2 ] =  1;
        b.geo.vol0      = -1;
        if( dr_0 > 0 )
        {
            b.geo.sfc0      [ 2 ] =    -1;
            b.geo.is_uniform[ 0 ] = false;
        }
        return;
    };
/*
    virtual void set_geo_fulldim( mesh::block_t & b )
    {
        auto & geo( b.geo );
        for( auto i : utils::loop( geo.dxf[ 1 ].n_cell ) )
            geo.dxf[ 1 ]( i ) = geo.x_cc( 0, i[ 0 ] )
                            * ( geo.x_fc( 1, i[ 1 ] + 1 ) -
                                geo.x_fc( 1, i[ 1 ]   ) ) ;
        return base_t::set_geo_fulldim( b );
        };  */
};

};// namespace mesh::meshgen
