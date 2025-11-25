#include "geometry.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Mesh geometry functions

__host__ geo_t::geo_t(  )
{
    for( int a = 0; a < 3; ++ a )
    {
        is_uniform[ a ]     =  true ;
        sfc0      [ a ]     =     1 ;
        dx0       [ a ]     =     1 ;
        xf0       [ a ]     =     1 ;
        for( int m = 0; m < 2; ++ m )
        {
            refine_flag[ a ][ m ] = false;
            coarse_flag[ a ][ m ] = false;
            phybnd_flag[ a ][ m ] = false;
        }
        xv[ a ] = nullptr;
        xf[ a ] = nullptr;
    }
    vol0 = 1;
    return;
}

__host__ void geo_t::setup( const   idx_t & n_ceff ,
         const int  & n_gh, const f_new_t &  f_new )
{
    this->n_ceff = n_ceff;
    this->n_gh   =   n_gh;
    this->n_dim  =      0;
    for( int a = 0; a < 3; ++ a )
    {
        if( ! is_uniform[ a ] )
        {
            auto s_cell = type::size( xv [ 0 ] ) *
                        ( n_ceff[ a ] + 2 * n_gh + 1 );
            xv [ a ]  = ( float_t * ) f_new ( s_cell );
            xf [ a ]  = ( float_t * ) f_new ( s_cell );
            dx0[ a ]  = -1;
        }
        n_dim  +=   ( n_ceff[ a ] > 1 ? 1 : 0 );
        auto n_face = n_ceff;
        ++   n_face[ a ];
        if( sfc0[ a ] <= 0 ||
            ( ! ( is_uniform[ ( a + 1 ) % 3 ] &
                  is_uniform[ ( a + 2 ) % 3 ] ) ) )
        {
            vol0       = -1;
            sfc0 [ a ] = -1;
            sfc  [ a ].init( f_new, n_face );
        }
    }
    if( vol0 <= 0 )
        vol.init( f_new, n_ceff );
    return;
}

__host__ void geo_t::setup( const   geo_t &   src ,
                            const f_new_t & f_new )
{
    return setup( src.n_ceff, src.n_gh, f_new );
}

__host__ void geo_t::free( const f_free_t & f_free )
{
    for( int a = 0; a < n_dim; ++ a )
    {
        f_free       (  xv[ a ] );
        f_free       (  xf[ a ] );
        sfc[ a ].free(   f_free );
    }
    vol.free( f_free );
    return;
}

__host__ void geo_t::cp_prim( geo_t & tgt ) const
{
    tgt.n_ceff = this->n_ceff;
    tgt.n_dim  = this-> n_dim;
    tgt.n_gh   = this->  n_gh;
    tgt.vol0   = this->  vol0;    
    for( int a = 0; a < 3; ++ a )
    {
        tgt.is_uniform[ a ] = this->is_uniform[ a ];
        tgt. dx0      [ a ] = this-> dx0      [ a ];
        tgt. xf0      [ a ] = this-> xf0      [ a ];
        tgt.sfc0      [ a ] = this->sfc0      [ a ];
        for( const auto & m : { 0, 1 } )
        {
            tgt.refine_flag[ a ][ m ]
              = refine_flag[ a ][ m ];
            tgt.coarse_flag[ a ][ m ]
              = coarse_flag[ a ][ m ];
            tgt.phybnd_flag[ a ][ m ]
              = phybnd_flag[ a ][ m ];
        }
    }
    return;
}

__host__ void geo_t::cp_shlw( geo_t & tgt ) const
{
    for( int a = 0; a < 3; ++ a )
    {
        tgt.xv [ a ] = xv [ a ];
        tgt.xf [ a ] = xf [ a ];
        tgt.sfc[ a ] = sfc[ a ];
    }
    tgt.vol = vol;
    return cp_prim( tgt );
} // Shallow copy

__host__ void geo_t::cp_deep
( geo_t & tgt, const f_cp_t & f_cp ) const
{
    for( int a = 0; a < 3; ++ a )
    {
        const auto s_cp = type::size( xv[ 0 ] )
             * ( n_ceff[ a ] + 2 * n_gh + 1 ) ;
        if( dx0[ a ] <= 0 )
        {
            f_cp( tgt.xv[ a ], xv[ a ], s_cp );
            f_cp( tgt.xf[ a ], xf[ a ], s_cp );
        }
        if( sfc[ a ] )
            sfc[ a ].cp_to( tgt. sfc[ a ], f_cp );
    }
    if( vol0 <= 0 )
        vol.cp_to( tgt.vol, f_cp );
    return;
}

__host__ void geo_t:: read
( const f_new_t & f_new, const f_read_t & f_r )
{
    f_r  ( xf0,            "xf0" );
    f_r  ( dx0,            "dx0" );
    f_r  ( sfc0,          "sfc0" );
    f_r  ( & vol0,        "vol0" );
    f_r  ( & n_gh,        "n_gh" );
    f_r  ( n_ceff.x,    "n_ceff" );
    f_r  ( is_uniform, "uniform" );    
    setup( n_ceff, n_gh,   f_new );
    for( int a = 0 ; a < 3; ++ a )
        if( ! is_uniform[ a ] )
        {
            f_r( xv[ a ], "xv_" + std::to_string( a ) );
            f_r( xf[ a ], "xf_" + std::to_string( a ) );
        }
    return;
}

__host__ void geo_t::write( const f_write_t & f_w_ ) const
{
    auto f_w = [ & ]( auto * p, size_t c, std::string tag )
    {    f_w_( ( void * )p , c, type::size( p ),  tag );  };
    f_w( xf0,         3,      "xf0" );
    f_w( dx0,         3,      "dx0" );
    f_w( sfc0,        3,     "sfc0" );
    f_w( & vol0,      1,     "vol0" );
    f_w( & n_gh,      1,     "n_gh" );
    f_w( n_ceff.x,    3,   "n_ceff" );
    f_w( is_uniform,  3,  "uniform" );    
    for( int a = 0; a < 3; ++ a )
        if( ! is_uniform[ a ] )
        {
            const auto axis = std::to_string       ( a );
            const auto len  = 2 * n_gh + 1 + n_ceff[ a ];
            f_w( xv[ a ], len, "xv_" + axis );
            f_w( xf[ a ], len, "xf_" + axis );
        }
    return;
}
};                              // namespace mesh
