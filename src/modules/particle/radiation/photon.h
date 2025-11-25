#pragma once

#include "../../../utilities/types/crtp.h"
#include "../types/particle_base.h"

namespace particle::radiation::photon
{
////////////////////////////////////////////////////////////
// Basic photon type in Cartesian coordinates

template< class derived_T = crtp::   dummy_t >
struct cart_t : public  particle::par_base_t
{
    ////////// Types //////////
    __crtp_def_self__( cart_t, derived_T );
    using geo_loc_t  = particle::geo_loc_t;
    int   id;

    ////////// Functions //////////
    template < class x_T, class prx_T >
    __device__ __forceinline__  int get_dl
    ( x_T & dl, geo_loc_t & g_l, const prx_T & prx )
    {
        for( int a = 0; a < dir.size(  ); ++ a )
        {
            if( dir[ a ] == 0 )
            {
                dl [ a ] = -1 ;
                continue;
            }
            const auto & x_f = dir[ a ] > 0 ? g_l.x_r[ a ]
                                            : g_l.x_l[ a ];
            dl[ a ]  = ( x_f - x  [ a ] )   / dir    [ a ];
        }
        int           a_proc (     0   ) ;
        type::float_t dl_min ( FLT_MAX ) ;
        for( int a = 0; a < dl.size(   ) ;  ++   a )
            if( dl[ a ] >= 0 && dl [ a ] <= dl_min )
            {
                a_proc  =       a;
                dl_min  = dl[ a ];
            }
        return a_proc;
    };

    template < class f_T > __device__ __forceinline__ bool
    sim_eq( const f_T & a, const f_T & b, const f_T & th )
    {
        if( a == 0 )
            return b == 0;
        return utils::abs( ( a - b ) / a ) < th;
    };

    template < class x_T, class prx_T, class itg_T >
    __device__ __forceinline__ void proc_geo
    ( const int & a_proc , const   x_T &  dl ,
      geo_loc_t &    g_l , const prx_T & prx,
      const itg_T &  itg )
    {
        for( int a = 0; a < dl.size(  ); ++ a )
            if( a == a_proc )
            {
                const bool rhs( dir[ a ] > 0 );
                x[ a ]  = rhs ? g_l.x_r[ a ] : g_l.x_l[ a ];
                i[ a ] += rhs ? 1 : -1;
                g_l.shift( * this, prx.geo, a, rhs );
            }
            else
                x[ a ] += dl[ a_proc ] * dir[ a ];
        return;
    };

    template < class x_T, class prx_T, class itg_T >
    __device__ __forceinline__  void proc_phys
    ( const int       & a_proc , const   x_T &  dl,
      const geo_loc_t &    g_l , const prx_T & prx,
      const itg_T     &    itg )
    {
        const auto dsi = dl[ a_proc ] / prx.geo.volume( i );
        const auto flx = proper  * dsi;
        atomicAdd( prx.rad.flx.at( i ), flx );
        return;
    };

    template < class bmp_T, class itg_T >
    __device__ __forceinline__ bool proc_step
    ( type::coord_t &   dl, geo_loc_t   & g_l ,
      const   bmp_T & bmap, const itg_T & itg )
    {
        const  auto & prx( bmap [ ib_l ] );
        auto & self = get_self_c(  );
        const  auto a_proc  = self.get_dl(  dl, g_l, prx );
        self.  proc_phys    ( a_proc,  dl, g_l, prx, itg );
        self.  proc_geo     ( a_proc,  dl, g_l, prx, itg );
        const  auto proc_flag = prx.shift_blk     ( self );
        if( proc_flag & self.regulate( bmap ) )
        {
            g_l.load( bmap[ ib_l ].geo,      i );
            g_l .fix( bmap[ ib_l ].geo, * this );
        }
        return proc_flag;
    };
    
    template < class map_T, class itg_T > __device__
    int proc( const map_T & bmap, const itg_T & itg )
    {
        auto & self = get_self_c(  );
        this->dest.i_rank = -1;
        geo_loc_t    g_l ;
        type::coord_t dl ;
        for( int  ib_old = -1; step > 0; -- step )
        {            
            if( ib_l != ib_old && ib_l >= 0 )
            {
                g_l.load( bmap[ ib_l ].geo,      i );
                g_l.fix ( bmap[ ib_l ].geo, * this );
                ib_old = ib_l;
            }
            if( ! self.proc_step( dl, g_l, bmap, itg ) )
                break;
        }
        if( step <= 0 )
            self.dest.todo = to_rm;
        return self.dest.i_rank;
    };
};
};                // namespace particle::radiation::photon
