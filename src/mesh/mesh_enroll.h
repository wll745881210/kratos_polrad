#pragma once

#include "mesh.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Block type

template< class blk_T >
void mesh_t::enroll_block(  )
{
    f_blk_yield = [ & ] (  )
    {
        return std::make_shared< blk_T > (  );
    };
    return;
}

template< class tre_T >
void mesh_t::enroll_tree(  )
{
    p_tre = std::make_shared< tre_T > (  );
    return;
}

////////////////////////////////////////////////////////////
// Computing modules

template < class mgn_T >
std::shared_ptr< mgn_T > mesh_t::enroll_mgen     (  )
{
    auto p = std::make_shared< mgn_T > (  );
    p_mgn  = p;
    return   p;
}

template < class com_T >
std::shared_ptr< com_T > mesh_t::enroll_comm     (  )
{
    auto  p  = std::make_shared< com_T > (  );
    if( ! p_dev )
        throw std::runtime_error
            ( "comm must be enrolled after device." );
    p->p_dev = p_dev;
    p_com    = p;
    return p ;
}

template < class cyc_T >
std::shared_ptr< cyc_T > mesh_t::enroll_cycle    (  )
{
    auto p = std::make_shared< cyc_T > (  );
    p_cyc  = p ;
    return p ;
}

template < class dev_T >
std::shared_ptr< dev_T > mesh_t::enroll_device   (  )
{
    auto p = std::make_shared< dev_T > (  );
    p_dev  = p ;
    return p ;
}

template < class bio_T >
std::shared_ptr< bio_T > mesh_t::enroll_binary_io(  )
{
    auto p = std::make_shared< bio_T > (  );
    p_bio  = p ;
    return p ;
}

////////////////////////////////////////////////////////////
// Physical module enrollment for class mesh_t

template < class mod_T >
std::shared_ptr< mod_T > mesh_t::enroll_module
( const int &   i_init , const int   &  i_step )
{
    auto   p_mod    = std::make_shared< mod_T > (  );
    steps[ i_step ] = [ &, p_mod ] (  )
    {
        p_mod->step( * this );
    };
    reads.push_back ( [ &, p_mod ] ( const input & args )
    {
        p_mod->read( args );
        order_max = std::max
                ( p_mod->space_order(  ), order_max );
    }   );
    inits[ i_init ] = [ &, p_mod ] ( const input & args )
    {
        p_mod->init( args, * this );
    };
    p_mod->p_mesh   =  this  ;
    mods . push_back( p_mod );
    return p_mod ;
}

template < class mod_T >
std::shared_ptr< mod_T > mesh_t::enroll_module(  )
{
    auto it_ini = inits.rbegin(   );
    auto it_stp = steps.rbegin(   );
    return enroll_module  < mod_T >
    ( it_ini == inits.rend(  )  ? 0 : it_ini->first + 1 ,
      it_stp == steps.rend(  )  ? 0 : it_stp->first + 1 );
}

};                              // namespace mesh
