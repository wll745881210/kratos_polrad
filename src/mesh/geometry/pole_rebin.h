#pragma once

namespace mesh
{
////////////////////////////////////////////////////////////
// Rebin near the polar axis for spherical/cylindrical coord

template< class bdt_T, class hlp_T, class reb_T >
__global__ void ker_pole_rebin
( const bdt_T bdt, const hlp_T hlp, const reb_T reb ,
  const int  step, const int  side )
{
    return reb.pole_rebin( bdt, hlp, step, side );
}

struct rebin_t
{
    ////////// Type and data //////////
    using dual_t = mesh::block::dual_t;

    int             rebin_lvl;
    type::float_t th_lim[ 2 ];

    ////////// Functions //////////    
    __host__ virtual void read( const input & args )
    {
        rebin_lvl   = args.get< int >
            ( "mesh", "pole_rebin_lvl",     6 );
        th_lim[ 0 ] = args.get< type::float_t >
            ( "mesh", "pole_rebin_thlim", 0.5 );
        th_lim[ 1 ] = 3.141592654 - th_lim[ 0 ];
        return;
    };

    template< class geo_T >    
    __host__ __device__ __forceinline__ bool at_pole
    ( const geo_T & geo, const bool & rhs ) const
    {
        if( ! rhs )
            return geo.phybnd_flag[ 1 ][ 0 ] &&
                   geo.x_fc( 1, 0 ) < th_lim[ 0 ];
        return geo.phybnd_flag[ 1 ][ 1 ] &&
               geo.x_fc( 1, geo.n_ceff[ 1 ] ) > th_lim[ 1 ];
    };

    template< class geo_T, class i_T > __device__
    int num_rebin( const geo_T & geo, const i_T & i ) const
    {
        int n_rebin( 1 );
        int di_ths [ 2 ] = { -1, -1 };
        if( at_pole( geo, false ) )
            di_ths[ 0 ] =  i[ 1 ] ;
        if( at_pole( geo,  true ) )
            di_ths[ 1 ] = geo.n_ceff[ 1 ] - 1 - i[ 1 ];

        int & di_th( di_ths[ 0 ] );
        if( di_ths[ 1 ] >= 0 && di_ths[ 1 ] < di_th )
            di_th = di_ths[ 1 ];
        if( di_th >= 0 && di_th < rebin_lvl )
            n_rebin = utils::min( geo.n_ceff[ 2 ],  1 <<
                                ( rebin_lvl - di_th ) );
        return n_rebin;
    };    
    
    template < class bdt_T, class hlp_T > __device__ void
    pole_rebin( const bdt_T & b_d, const hlp_T & hlp,
                const int   &   s, const int & side ) const
    {
        const auto & u     = hlp.get_data_fld( b_d );
        const auto & n_var = u.n_int;

        using f_t = typename type::prim_type_t
                  < decltype( u ) >::f_t;
        
        type::idx_t i;
        if( side == 0 )
            i[ 1 ] = blockIdx.x;
        else
            i[ 1 ] = b_d.geo.n_ceff[ 1 ] - 1 - blockIdx.x;
        i[ 0 ] = threadIdx.x;
        const auto & n_rebin( num_rebin( b_d.geo, i ) );

        __dyn_shared__( char,     dat_sh );
        f_t  * tot =  ( f_t * ) ( dat_sh )
                   + n_var * threadIdx.x;
        int r( 0 );
        for( int j = 0; j < b_d.geo.n_ceff[ 2 ]; ++ j )
        {
            if( r >= n_rebin )
                r -= n_rebin ;
            if( r == 0 )
                for( int n = 0; n < n_var; ++ n )
                    tot[ n ] = 0;            

            i[ 2 ] = j;
            auto * u_src = u.at( i, s );
            for( int n = 0; n < n_var; ++ n )
                tot[ n ] += u_src[ n ];
            if( r == n_rebin - 1 )
            {
                for( int n = 0; n < n_var;   ++ n )
                    tot[ n ] /= n_rebin;
                for( int l = 0; l < n_rebin; ++ l )
                {
                    i[ 2 ] = j - l;
                    auto * u_tgt = u.at( i, s );
                    for( int n = 0; n < n_var; ++ n )
                        u_tgt[ n ] = tot[ n ];
                }
            }
            ++ r;
        }
        return;        
    };

    template< class bdt_T, class hlp_T > __host__ void
    operator(  ) ( const bdt_T & b_d, const dual_t  & d,
                   const hlp_T & hlp, const int &  step,
                   const device::base_t & dev ) const
    {
        if( rebin_lvl < 1 )
            return;
        const auto & u     = hlp.get_data_fld( b_d );
        const auto & n_var = u.n_int;
        using f_t = typename type::prim_type_t
                  < decltype( u ) >::f_t;
        const int n_bl = utils::min
            ( rebin_lvl, b_d.geo.n_ceff[ 1 ] );
        const int n_th = b_d.geo.n_ceff[ 0 ];
        const int n_sh = n_var * sizeof( f_t ) * n_th;
        const auto rsc = std::make_tuple
            ( dim3( n_bl ), dim3( n_th ), n_sh );

        if( at_pole( b_d.geo, false ) )
            dev.launch( ker_pole_rebin, rsc, d.stream,
                        b_d, hlp, * this, step, 0 );
        if( at_pole( b_d.geo,  true ) )
            dev.launch( ker_pole_rebin, rsc, d.stream,
                        b_d, hlp, * this, step, 1 );
        return;
    };
};

};                              // namespace mesh
