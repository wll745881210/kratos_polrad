#pragma once

namespace radiation::compress
{
using ::radiation::photon::geo_loc_t;

////////////////////////////////////////////////////////////
// Compressed photon type

template< class par_T >
struct par_t          // "Compressed" particle storage type
{
    ////////// Data //////////
    type::uchar_t      lum;
    type::uchar_t     iblk;
    type::uchar_t    steps;    
    type::uchar_t   i[ 4 ];
    type::uchar_t  dx[ 3 ];
    type::uchar_t dir[ 2 ];    // mu, phi

    ////////// Constants //////////
    static const constexpr type::float_t uchar_m
    = std::numeric_limits< type::uchar_t >:: max(  );

    ////////// Conversion //////////
    template< class map_T >
    __forceinline__ __host__ __device__ void load
    ( par_T & p, geo_loc_t & g_l, const map_T & bmap ) const
    {
        const auto uchar_mi( 1 / uchar_m );
        // Indices and coordinates
        p.ib_l     =   iblk;
        for( int a = 0; a < 3; ++ a )
        {
            p.i[ a ]  =   i[ a ] ;
            p.i[ a ] |= ( i[ 3 ] >> ( 2 * a ) ) << 8;
        }
        g_l.load ( bmap[ iblk ].geo, p.i );
        for( int a = 0; a < 3 ;  ++  a )        
            p.x[ a ]    =   g_l.x_l[ a ] + dx     [ a ] *
               uchar_mi * ( g_l.x_r[ a ] - g_l.x_l[ a ] );

        short i_lum =  lum;
        i_lum      |= ( i[ 3 ] >> 6 ) << 8 ;
        p.to_proper( i_lum / ( type::float_t ) ( 1023 ) );
        p.step_remain      =   steps;
        
        // Direction
        const auto conv = uchar_mi * 2  * 3.14159f ;
        const auto mu   = 2 * dir[ 0 ]  * uchar_mi - 1;
        const auto nu   = __sqrt1( 1    - mu  *   mu );
        sincosf( dir[ 1 ] * conv, p.dir + 1, p.  dir );
        p.dir[ 0 ] *= nu;
        p.dir[ 1 ] *= nu;
        p.dir[ 2 ]  = mu;
        return ;
    };
    
    __forceinline__ __device__ __host__
    void save( const par_T & p, const geo_loc_t & g_l )
    {
        // Indices and coordinates
        steps  = p.step_remain;
        iblk   = p.       ib_l;
        type ::uchar_t  j ( 0 );
        for( int a = 0; a < 3 ; ++ a )
        {
            i [ a ]   =   p.i    [ a ] ;
            i [ 3 ]  |= ( p.i    [ a ] >> 8 ) << ( 2 * a );
            dx[ a ]   = ( p.x    [ a ] - x0      [ a ] ) *
              uchar_m / ( g_l.x_r[ a ] - g_l.x_l [ a ] ) ;
        }
        const short i_proper = p.from_proper(  ) * 1023;
        lum   =     i_proper ;
        j    |=   ( i_proper >> 8 ) << 6;
        
        // Direction
        const auto phi = atan2f( p.dir[ 1 ], p.dir[ 0 ] );
        dir [ 1  ] = uchar_m * ( p.dir[ 2 ]     + 1 ) / 2;
        dir [ 0  ] = uchar_m * ( phi / 3.14159f + 1 ) / 2;
        return;
    };

    ////////// Process //////////
    template< class map_T, class com_T > __forceinline__
    __device__ int proc( const map_T & bmap )
    {
        par_T       p;
        geo_loc_t g_l;
        this->load( p, g_l, bmap );
        const auto i_rank = p.proc( bmap, comm );
        if( i_rank >= 0 )
            this->save( p , g_l );
        return  i_rank;
    };
};
};
