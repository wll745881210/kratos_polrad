#pragma once

#include "cylindrical.h"

namespace mesh::meshgen
{
////////////////////////////////////////////////////////////
// Cartesian unifor mesh generator

class sph_t : public cyl_t
{
    ////////// Spherical geometry //////////
public:
    virtual float2_t surface
    ( const idx_t &  i , const mesh::geo_t & geo,
      const int   & ax ) const
    {
        const auto   r_l = geo.x_fc( 0, i[ 0 ]     );
        const auto   r_r = geo.x_fc( 0, i[ 0 ] + 1 );
        const auto  th_l = geo.x_fc( 1, i[ 1 ]     );
        const auto  th_r = geo.x_fc( 1, i[ 1 ] + 1 );
        const auto d_phi = geo.x_fc( 2, i[ 2 ] + 1 )
                         - geo.x_fc( 2, i[ 2 ]     );
        if( ax == 0 )
            return pow( r_l, 2 )  *  d_phi
               * fabs( cos( th_l ) - cos( th_r ) );
        const auto dr2 = pow( r_r, 2 ) - pow( r_l, 2 );
        if( ax == 1 )
            return dr2 / 2 * sin( th_l ) * d_phi;
        return dr2 / 2 * ( th_r - th_l );
    };
    virtual float2_t volume
    ( const idx_t & i, const mesh::geo_t & geo ) const
    {
        const auto   r_l = geo.x_fc( 0, i[ 0 ]     );
        const auto   r_r = geo.x_fc( 0, i[ 0 ] + 1 );
        const auto  th_l = geo.x_fc( 1, i[ 1 ]     );
        const auto  th_r = geo.x_fc( 1, i[ 1 ] + 1 );
        const auto d_phi = geo.x_fc( 2, i[ 2 ] + 1 )
                         - geo.x_fc( 2, i[ 2 ]     );
        return ( pow( r_r, 3. ) - pow( r_l, 3. ) ) / 3.
             * fabs( cos( th_l ) - cos( th_r ) ) * d_phi;
    };

    virtual void set_uniformity ( block_t & b )
    {
        for( int a = 0; a < 3; ++ a )
        {
            b.geo.is_uniform[ a ] = true;
            b.geo.sfc0      [ a ] =    -1;
        }
        b.geo.vol0 = -1;
        // b.geo.is_uniform[ 2 ] = true;
        if( dr_0 > 0 )
            b.geo.is_uniform[ 0 ] = false;
        return;
    };

    virtual void set_coord_axes( block_t & b )
    {
        auto & geo( b.geo );
        auto & reg( b.reg );

        for( int a = 0; a < 3; ++ a )
        {
            auto loc_f = [ & ] ( float2_t i ) -> float2_t
            {
                if( geo.n_ceff[ a ] == 1 )
                    return i; // [0, 1) Even at higher level
                i  = ( reg[ a ] + i / geo.n_ceff[ a ] )
                    / ( i_logic_lim[ a ]  << reg.level );
                return location( i, a );
            };
            auto loc_c = [ & ] ( float2_t i ) -> float2_t
            {
                if( geo.n_ceff[ a ] == 1 )
                    return i; // [0, 1) Even at higher level
                if( a == 0 )
                {
                    const auto r_r( loc_f( i + 1 ) );
                    const auto r_l( loc_f( i     ) );
                    return 3. / 4.
                        * ( pow( r_r, 4 ) - pow( r_l, 4 ) )
                        / ( pow( r_r, 3 ) - pow( r_l, 3 ) );
                }
                if( a == 1 )
                {
                    const auto th_r = loc_f( i + 1 );
                    const auto th_l = loc_f( i     );
                    float2_t res = th_r - th_l
                        + cos( th_l ) * sin( th_l )
                        - cos( th_r ) * sin( th_r );
                    res /=
                        2 * ( cos( th_l ) - cos( th_r ) );
                    return res;
                }
                return loc_f( i + 0.5 );
            };
            geo.xf0[ a ] = loc_f( 0 );
            if( geo.is_uniform[ a ] )
            {
                geo.dx0[ a ] = loc_f( 1 ) - geo.xf0[ a ];
                continue;
            }
            else
                geo.dx0[ a ] = -1;
            auto *  xf =  geo.xf[ a ] + geo.n_gh;
            auto *  xv =  geo.xv[ a ] + geo.n_gh;
            for( int i = -geo.n_gh;
                 i <= geo.n_ceff[ a ] + geo.n_gh; ++ i )
            {
                xf[ i ]  = loc_f( i );
                xv[ i ]  = loc_c( i );
            }
        }
        return;
    };  
/*    
    virtual void set_geo_fulldim( mesh::block_t & b )
    {
        cyl_t::set_geo_fulldim( b );
        auto & geo  ( b.h(  ).geo );
        geo.dx0[ 2 ] = -1;
        for( auto i : utils::loop( geo.dxf[ 2 ] .n_cell ) )
            geo.dxf[ 2 ]( i ) = geo.x_cc( 0, i[ 0 ] )
                        * sin ( geo.x_cc( 1, i[ 1 ] ) )
                            * ( geo.x_fc( 2, i[ 2 ] + 1 ) -
                                geo.x_fc( 2, i[ 2 ] ) ) ;
        return;
    };
p*/
};

};// namespace mesh::meshgen
