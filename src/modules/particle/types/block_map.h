#pragma once

#include "driver_base.h"
#include "particle_base.h"
#include "proxy.h"
#include "../driver.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Mapping the local ID to the actual grid

template< class prx_T >
struct block_map_t  : public driver::base_t
{
    ////////// Type //////////
    using prx_t =                 prx_T;    
    using bdt_t = typename prx_t::bdt_t;

    ////////// Data //////////
    bool       periodic[ 3 ];
    bool           is_static;
    int                n_prx;
    prx_T             *  prx;
    type::float_t          t;
    type::float_t         dt;
    type::float_t     * p_dt;
    type::float_t     dt_lim;
    type::coord_t      l_box;    
    type::coord_t  xlim[ 2 ];

    ////////// Functions //////////
    __host__ block_map_t(  ) 
           : prx ( nullptr ), n_prx( 0 ) {  };

    __host__ virtual void init
    ( const  input & args, particle::base_t & mod )
    {
        is_static = args.get< bool >
                  ( "particle", "static_proxy", false );
        return driver::base_t::init( args, mod );
    };

    __host__ virtual void set_dt( particle::base_t & mod )
    {
        t      = mod.p_mesh->p_cyc->t;
        dt     = mod.p_mesh->p_cyc->dt(  );
        p_dt   = mod.p_mesh->p_cyc->p_dt;
        dt_lim = mod.p_mesh->p_cyc->dt_expand * dt;
        return;
    };

    __host__ virtual void  update
    ( const mesh::mod_base_t::v_reg_t & news ,
      particle  ::             base_t &  mod )
    {   //  v-- n_prx > 0 means called more than once
        if( is_static && n_prx > 0 )
            throw std::runtime_error( "static prx w AMR" );
        const auto rank_this( mod.p_bdk->p_com->rank(  ) );
        std::map < int, int > m_rank;
        for( auto & b : ( * mod. p_mesh ) )
            for( const auto & [ nb, reg ] : b.neighbors )
                if( nb.guest_info.rank  !=    rank_this )
                    m_rank[ nb.guest_info.rank ] = -1;

        const auto & tree( mod.p_mesh->tree(  ) );
        for( int a = 0; a < 3; ++ a )
            periodic[ a ] = tree.periodic[ a ];
        
        int  count = 0 ;
        for(  auto & [ rank, i_rank ] : m_rank )
            i_rank = ( count ++ );

        // Local proxies
        n_prx = mod. p_blk_local->size(   );
        std::vector< prx_T > prx_h( n_prx );
        for( auto & d : mod )
            prx_h[ d.b_w.get(  ).id_l ].setup
                 ( d, m_rank, mod );

        auto  &  f_dev( * mod . p_dev ) ;
        const auto s_p( n_prx * sizeof( prx_T ) );
        if( is_static )
        {
            prx = ( prx_T * ) f_dev.f_malloc_const ( s_p );
            f_dev.  f_cc    ( prx,  prx_h. data(  ), s_p );
        }
        else
        {
            if( prx !=  nullptr )
                f_dev.free_device( prx );
            prx = ( prx_T * ) f_dev.f_malloc( s_p );
            f_dev.f_cp( prx , prx_h.data(  ), s_p );
        }
        auto & xlim = mod.p_mesh->p_mgn->x_lim_global;
        for( int a = 0; a < 3; ++ a )
        {
            for( int l = 0; l < 2; ++ l )
                 this -> xlim[ l ][ a ] = xlim[ l ][ a ];
            l_box[ a ] = xlim[ 1 ][ a ] - xlim[ 0 ][ a ];
        }
        return f_dev.sync_all_streams(  );
    };
    __host__ virtual void finalize( particle::base_t & mod )
    {
        if( ! is_static && prx != nullptr )
            mod.p_dev->free_device  ( prx );
        return;
    };

    __device__ prx_T & operator [  ] ( const int & i ) const
    {
        return prx[ i ];
    };
};
};                              // namespace particle
