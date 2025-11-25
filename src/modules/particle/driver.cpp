#include "particle.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Initialization and destruction

void base_t::init( const input & args, mesh::mesh_t & mesh )
{
    if( ! p_bdk )
    {
        struct keeper_t : public mesh::boundary::keeper_t
        {
            virtual void update( mesh::mesh_t & ) {  };
        };
        enroll_bdk< keeper_t > (  );
    }   // The particles are transported in their own way.
    n_cyc = args.get< int > ( "particle", "n_cycle_stp", 1 );
    p_dev      = mesh. p_dev ;
    p_bdk->init( args, mesh );
    for( auto & p : drv )  // sub-module init needs p_dev
         p->init( args, * this );
    stream = mesh.p_dev->yield_stream(  );
    event  = mesh.p_dev->event_create(  );
    return update( mesh ); // mesh init updates need sub-mod
}

void base_t::finalize( mesh::mesh_t & mesh )
{
    for( auto  & p   :  drv )
        p->finalize( * this );
    mesh:: mod_base_t::finalize(  mesh );
    return p_dev->event_destroy( event );
}

void base_t::save( mesh     ::mesh_t & mesh,
                   binary_io::base_t & bio )
{
    mod_base_t   ::save( mesh, bio );
    return   f_par_save( mesh, bio );
}

void base_t::update
( const mod_base_t::v_reg_t & news, mesh::mesh_t & mesh )
{
    mod_base_t::update( news,   mesh );
    for( auto  & p :    drv )
        p->update     ( news, * this );
    return;
}

////////////////////////////////////////////////////////////
// Interfaces

void base_t::step( mesh::mesh_t & mesh )
{
    for( int c = 0 ; c < n_cyc; ++ c )
        integrate( mesh, c );
    return p_dev->sync_stream( this->stream );
}

int  base_t::space_order(  )
{
    return 1;
}

comm_mode_t base_t::comm_mode(  )
{
    return refresh_pool;
}

};                              // namespace particle
