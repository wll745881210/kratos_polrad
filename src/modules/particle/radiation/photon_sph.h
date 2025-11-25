#pragma once

#include "photon.h"

namespace particle::radiation::photon
{
////////////////////////////////////////////////////////////
// Basic photon type in Cartesian coordinates

template< class derived_T = crtp::dummy_t >
struct sph_t : cart_t< crtp::helper< sph_t, derived_T > >
{
    ////////// Types //////////
    __crtp_def_self__( sph_t, derived_T );
    using geo_loc_t = particle::geo_loc_t;
    using   float_t = type    ::  float_t;
    using   super_t
            = cart_t< crtp::helper< sph_t, derived_T > >;

    type::idx_t idx_dir;    
    coord_t       x_car;

    ////////// Functions //////////
    template< class g_T, class f_T > __device__
    bool root( g_T & x_1, g_T & x_2, const f_T & a,
               const f_T & b, const f_T & c ) const 
    {
        const auto det = b * b - 4 * a * c;
        if( det < 0 )
        {
            x_1    = det;
            return false;
        }
        const auto nom = copysignf( sqrtf( det ), b ) + b;
        const auto x0  = nom / ( -2 * a );
        const auto x1  = ( -2 * c ) / nom;
        x_1 = utils::min ( x0, x1 ) ;
        x_2 = utils::max ( x0, x1 ) ;
        return true;
    };    
    
    template < class x_T, class prx_T >
    __device__ __forceinline__  int get_dl
    ( x_T & dl, geo_loc_t & g_l, const prx_T & prx )
    {        
        const auto & geo( prx.geo );
        auto  &  x  ( this->x     );        
        auto  &  dir( this->  dir );

        coord_t theta_hat;
        {
            const auto cos_th =        cosf( x[ 1 ] );
            theta_hat[ 0 ] =  cos_th * cosf( x[ 2 ] );
            theta_hat[ 1 ] =  cos_th * sinf( x[ 2 ] );
            theta_hat[ 2 ] =         - sinf( x[ 1 ] );
        }
        float_t x_dot_d( 0 ), dt_dl( 0 );
        for( int a = 0; a < geo.n_dim; ++ a )
        {
            x_dot_d +=     x_car[ a ] * dir[ a ];
            dt_dl   += theta_hat[ a ] * dir[ a ];
        }
        const auto x2 = x[ 0 ] * x[ 0 ];
        const auto dr_dl = x_dot_d;
        const auto df_dl = x_car[ 0 ] * dir[ 1 ]
                         - x_car[ 1 ] * dir[ 0 ];

        idx_dir[ 0 ] = utils::sgn( dr_dl );
        idx_dir[ 1 ] = utils::sgn( dt_dl );
        idx_dir[ 2 ] = utils::sgn( df_dl );

        coord_t x_fwd, x_bck;
        for( int a = 0; a < 3; ++ a )
        {
            x_fwd[ a ] = ( idx_dir[ a ] > 0 ?
                           g_l.x_r[ a ] : g_l.x_l[ a ] );
            x_bck[ a ] = ( idx_dir[ a ] < 0 ?
                           g_l.x_r[ a ] : g_l.x_l[ a ] );
        }

        float_t x_1, x_2;
        
        ////////////////////////////////////////
        // r-direction
        float_t a_r =  1 ;
        float_t b_r =  2 * x_dot_d;
        float_t c_r = x2 - x_fwd[ 0 ] * x_fwd[ 0 ];
        auto real_root = root( x_1, x_2, a_r, b_r, c_r );
        
        if( ! real_root )
        {
            c_r =  x2 - x_bck[ 0 ] * x_bck[ 0 ];
            real_root = root( x_1, x_2, a_r, b_r, c_r );
            dl     [ 0 ]  = x_2;
            idx_dir[ 0 ] *=  -1;
        }   // Getting out from the same surface
        else
            dl[ 0 ] = ( x_1 > 0 ? x_1 : x_2 );

        ////////////////////////////////////////
        // theta-direction
        if( geo.n_dim > 1 )
        {
            const auto &  z = x_car[ 2 ];
            const auto & dz = dir  [ 2 ];

            auto mu2 = cosf( x_fwd[ 1 ] );
            mu2      = mu2 * mu2;
            auto a_m = dz * dz - mu2;
            auto b_m = 2 * ( z * dz - mu2 * x_dot_d );
            auto c_m = z * z - mu2 * x2;
            if( a_m != 0 )
            {
                real_root = root( x_1, x_2, a_m, b_m, c_m );
                if( ! real_root )
                {
                    mu2 = cosf( x_bck[ 1 ] );
                    mu2 = mu2 * mu2;
                    a_m = dz * dz - mu2;
                    b_m = 2 *( z * dz - mu2 * x_dot_d );
                    c_m = z * z - mu2 * x2;
                    if( a_m != 0 )
                    {
                        real_root = root
                            ( x_1 , x_2, a_m, b_m, c_m );
                        dl[ 1 ] = ( real_root ? x_2 : -1 );
                    }
                    else
                        dl [ 1 ]  = -c_m / b_m;
                    idx_dir[ 1 ] *= -1;
                }
                else
                    dl[ 1 ] = ( x_1 > 0 ? x_1 : x_2 );
            }
            else
                dl[ 1 ] = -c_m / b_m;
        }

        ////////////////////////////////////////
        // phi-direction
        if( geo.n_dim > 2 )
        {
            const auto sin_ph = sinf( x_fwd[ 2 ] );
            const auto cos_ph = cosf( x_fwd[ 2 ] );
            const auto a_f    = dir  [ 0 ] * sin_ph
                              - dir  [ 1 ] * cos_ph;
            const auto b_f    = x_car[ 1 ] * cos_ph
                              - x_car[ 0 ] * sin_ph;
            if( a_f != 0 )
                dl[ 2 ] = b_f / a_f;
            else
                dl[ 2 ] = -1;
            if( dl[ 2 ] < 0 )
            {
                const auto sin_ph = sinf( x_bck[ 2 ] );
                const auto cos_ph = cosf( x_bck[ 2 ] );
                const auto a_f    = dir  [ 0 ] * sin_ph
                                  - dir  [ 1 ] * cos_ph;
                const auto b_f    = x_car[ 1 ] * cos_ph
                                  - x_car[ 0 ] * sin_ph;
                if( a_f != 0 )
                    dl [ 2 ]  = b_f / a_f;
                else
                    dl [ 2 ]  = -1;
                idx_dir[ 2 ] *= -1;
            }
        }

        ////////////////////////////////////////
        // Find the direction
        int a_proc( 0 );
        type::float_t dl_min ( FLT_MAX );
        for( int a = 0; a < geo.n_dim; ++ a )
            if( dl [ a ] > 0 && dl [ a ] < dl_min )
            {
                a_proc  =       a;
                dl_min  = dl[ a ];
            }
        return a_proc;
    };

    __device__ __host__  void to_car(  )
    {
        const auto & x ( this->x );
        auto  &   x_car( this->x_car );        
        const auto s_th( sinf( x[ 1 ] ) );
        x_car[ 0 ] = x[ 0 ] * s_th * cosf( x[ 2 ] );
        x_car[ 1 ] = x[ 0 ] * s_th * sinf( x[ 2 ] );
        x_car[ 2 ] = x[ 0 ] *        cosf( x[ 1 ] );
        return;
    };

    __device__ __host__ __forceinline__ void to_sph(  )
    {
        auto & x( this->x );
        x[ 0 ] = this->x_car.norm(  );
        x[ 1 ] = x_car[ 2 ] / x [ 0 ];
        if( x[ 1 ] > 1 )
            x[ 1 ] = 0 ;
        else if( x[ 1 ] < -1 )
            x[ 1 ] = 3.141592654f;
        else if( x[ 1 ] < 1e-2f )
            x[ 1 ] = 1.570796f - x[ 1 ];
        else
            x[ 1 ] = acosf( x[ 1 ] );
        x[ 2 ] = atan2f( x_car[ 1 ] , x_car[ 0 ]  );
        return;
    };

    template< class glc_T > __device__ __host__
    void to_sph_guard( const glc_T & g_l )
    {
        static const float_t pi( 3.14159265358979323846 );
        this->to_sph(  );
        while( this->x[ 2 ] < g_l.x_l[ 2 ] - pi )
            this->x[ 2 ] += pi * 2;
        while( this->x[ 2 ] > g_l.x_r[ 2 ] + pi )
            this->x[ 2 ] -= pi * 2;
        if constexpr( std::is_same_v< float_t, float > )
            this->x[ 1 ] = __max1( this->x[ 1 ], 1e-3f );
        return;
    };
    
    template < class x_T, class prx_T, class itg_T >
    __device__ __forceinline__ void proc_geo
    ( const int & a_proc , const   x_T &  dl ,
      geo_loc_t &    g_l , const prx_T & prx,
      const itg_T &  itg )
    {
        static const float_t overshoot ( 1.001f );
        for( int a = 0; a < prx.geo.n_dim; ++ a )
            x_car[ a ] += dl[ a_proc ]
                       *  this->dir[ a ] * overshoot;
        this->to_sph_guard( g_l );
        this->i[ a_proc ] += idx_dir[ a_proc ] ;
        for( int a = 0; a < prx.geo.n_dim; ++ a )
            if( a != a_proc )
            {
                if( this->x[ a ] >= g_l.x_r[ a ] )
                    ++ this->i[ a ];
                else if( this->x[ a ] <= g_l.x_l[ a ] )
                    -- this->i[ a ];
            }
        return;
    };

    template < class bmp_T, class itg_T >
    __device__ __forceinline__ bool proc_step
    ( type::coord_t &   dl, geo_loc_t   & g_l ,
      const   bmp_T & bmap, const itg_T & itg )
    {
        const  auto & prx( bmap [ this->ib_l ] );
        auto & self = get_self_c(  );
        const  auto a_proc  = self.get_dl(  dl, g_l, prx );
        self.  proc_phys    ( a_proc,  dl, g_l, prx, itg );
        self.  proc_geo     ( a_proc,  dl, g_l, prx, itg );
        const  auto proc_flag = prx.shift_blk     ( self );
        if( proc_flag )
        {
            g_l.load( bmap[ this->ib_l ].geo, this->i );
            if( self.regulate( bmap ) )
                g_l .fix( bmap[ this->ib_l ].geo, * this );
        }
        return proc_flag;
    };    
};                // struct sph_t
};                // namespace particle::radiation::photon
