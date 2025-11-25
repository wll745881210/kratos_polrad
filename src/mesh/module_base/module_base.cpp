#include "module_base.h"
#include "../mesh.h"
#include "../boundary/keeper.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Default shapes of modules

mod_base_t::mod_base_t(  ) : need_update ( true ),
                             output_flag (    1 )
{
    p_blk_local = std::make_shared< b_map_t >(  );
    return;
}

void mod_base_t::init( const input & args, mesh_t & mesh )
{
    p_dev      = mesh. p_dev ;
    p_bdk->init( args, mesh );
    return update   (  mesh );
}

void mod_base_t::finalize( mesh_t & mesh )
{
    if( q_mod.expired(  ) )
        for( auto & [ reg, p ] : ( * p_blk_local ) )
            p->free (   );
    return;
}

void mod_base_t::parasite( std::weak_ptr< mod_base_t > q )
{
    p_blk_local = q.lock(  )->p_blk_local;
    need_update =  false;
    q_mod       =      q;
    return;
}

////////////////////////////////////////////////////////////
// Data access

block::dual_t &
mod_base_t::data  ( const region_logic_t & reg )
{
    return ( * ( * p_blk_local ) [ reg ] );
}

mod_base_t::p_ddt_t &
mod_base_t::p_data( const region_logic_t & reg )
{
    return     ( * p_blk_local ) [ reg ];
}

void mod_base_t::map_mesh( const mod_base_t::f_blk_t & f )
{
    for( auto & [ reg, p ] : ( * p_blk_local ) )
        f( reg, * p );
    return;
}

mod_base_t:: iter_t:: iter_t
( const  mod_base_t::b_map_t::iterator & it )
{
    static_cast< b_map_t::iterator & >( * this ) = it;
}

block::dual_t & mod_base_t:: iter_t::operator * (  )
{
    return ( * static_cast< b_map_t::iterator & >
           ( * this )->second );
}

mod_base_t::iter_t mod_base_t::begin(  )
{
    return   p_blk_local->begin(  );
}

mod_base_t::iter_t mod_base_t::  end(  )
{
    return   p_blk_local->  end(  );
}

////////////////////////////////////////////////////////////
// The update's that are in charge of mesh maintainance

void mod_base_t::update( mesh_t & mesh )
{
    if( ! q_mod.expired(  ) )
        return; // Return if this is a parasitic module
    const bool  mesh_changed( mesh.changed(  ) );
    if( ! ( need_update || mesh_changed ) )
        return;

    v_reg_t dels, news ;
    this->map_mesh( [ & ] ( auto & reg, auto & d )
    {
        if( ! mesh.contains( reg ) )
            dels.push_back ( reg ) ;
    }   );
    for( auto & b : mesh )
        if( p_blk_local->find( b.reg ) ==
            p_blk_local->end (     ) )
        {
            p_data( b.reg ) = block_yield( b,  mesh );
            auto    & geo ( data( b.reg ). p_d->geo );
            b.geo.cp_prim (    geo                  );
            geo.setup     ( b. geo, p_dev->f_malloc );
            b.geo.cp_deep (    geo, p_dev->f_cp     );
            news.push_back( b. reg  );
            data( b.reg ).p_h->geo  = b.geo;
        }

    if( mesh_changed )        // AMR  interface
        update( dels, news, mesh );
    else                      // Initialization
        update(       news, mesh );

    for( auto &  reg_del :  dels )
        p_blk_local->erase( reg_del );
    need_update =  false ;
    return p_bdk->update (  mesh );
}

void mod_base_t::update
( const mod_base_t::v_reg_t & news, mesh_t & mesh )
{
    p_dev->sync_all_streams(  );
    for( auto & reg : news )
        ( * p_blk_local )[ reg ]->setup(   );
    for( auto & reg : news )
    {
        auto p = ( * p_blk_local )[ reg ];        
        if( mesh. is_restart(   ) )
            p->read( * mesh.p_bio );
        else
            init_cond( * p );
        p-> copy_h2d (     );
    }
    return;
}

mod_base_t::p_ddt_t
mod_base_t::block_yield_base( block_t & b, mesh_t & mesh )
{
    if( ! ( f_bdt_yield && f_ddt_yield ) )
        throw std::runtime_error
            ( "Module has not have data type enrolled\n" );
    auto p = f_ddt_yield( b );
    p->p_d = f_bdt_yield( b );
    p->p_h = f_bdt_yield( b );
    p->set_dev( this->p_dev );
    p->order     = space_order  (   );
    p->p_h->p_dt = mesh.p_cyc->p_dt_h;
    p->p_d->p_dt = mesh.p_cyc->  p_dt;
    p->p_d->reg  =              b.reg;
    p->p_h->reg  =              b.reg;
    p->b_w       = std    :: ref( b );
    return p;
}

void mod_base_t::sync_streams( mesh_t & mesh )
{
    for( auto & d : ( * this ) )
        p_dev->sync_stream( d.stream );
    return;
}

void mod_base_t::save
( mesh_t & mesh, binary_io::base_t & bio )
{
    for( auto & d : ( * this ) )
         d.write( bio, output_flag );
    return;
}

};                              // namespace mesh
