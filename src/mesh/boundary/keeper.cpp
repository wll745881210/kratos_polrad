#include "../module_base/module_base.h"
#include "../mesh.h"
#include "keeper.h"

namespace mesh::boundary
{
////////////////////////////////////////////////////////////
// Initialization of boundary keeper

struct dummy_t : base_t
{
    int    side;
    auto & access_data( const block::dual_t & d )
    {
        return d.d(  );
    };
    __host__ void launch
    ( holder_p_t &, const int &, const device::base_t & )
    {  };
    template< class bdt_T > __device__ void operator (  )
    ( const bdt_T & b_d, const int & step ) const    {  };
};

keeper_t::keeper_t(  ) : n_th_lim( 32 ), p_mod( nullptr ),
                         independent_stream( true )
{
    enroll_phys< dummy_t > ( "dummy" );
    return;
}

void keeper_t::init( const input & args, mesh_t & mesh )
{
    independent_stream = args.get< bool >
         ( "boundary", "independent_stream", false );

    p_com = mesh . p_com->split(  ); // A different comm
    p_dev = mesh . p_dev;
    p_com->p_dev = p_dev;
    pool_phys.resize( 6 ); // Probably derived keeper done
    for( int s = 0; s < pool_phys.size(  ); ++ s )
        if( ! pool_phys[ s ].launch )
            pool_phys[ s ] = holder_phys[ "dummy" ];
    for( auto & f_init : inits )
        f_init  ( args ) ;
    return;
}

////////////////////////////////////////////////////////////
//

void keeper_t::assign_strm( holder_base_t & h )
{
    h.stream = ( ( ! independent_stream )  ? h.p_d->stream
                 : p_dev-> yield_stream (  ) );
    h. event = ( ( ! independent_stream )  ? h.p_d-> event
                 : p_dev-> event_create (  ) );
    return;
}

void keeper_t::assign_comm( block_t & b, mesh_t & mesh )
{
    if( ! b.updated_neighbor(  ) )
        return;

    auto & m_nb = bnd_comm[ b.reg ] ;
    m_nb.clear(  );
    for( const  auto & [ nb , reg ] : b.neighbors )        
    {
        if( ! desired_mode( nb ) )
            continue;
        auto & d ( p_mod->data( b.reg ) );
        auto & h = m_nb.emplace
             ( nb, holder_c_t( d ) ).first->second;
        f_comm_yield ( h );
        h.setup( nb, * p_mod );
        h.p_dev  = p_dev;
        assign_strm( h );
    }
    return;
}

void keeper_t::assign_phys( block_t & b, mesh_t & mesh )
{
    if( ! b.updated_neighbor(  ) )
        return;

    auto & mod  ( * p_mod ) ;    
    auto & m_ph = bnd_phys[ b.reg ] ;
    m_ph.clear  (   );
// #warning "Workaround; holder_c,p_t need move's"    
    m_ph.reserve( 6 );
    for( int a = 0; a < b.geo.n_dim; ++ a )
        for ( int i = 0; i < 2; ++ i )
            if( b.geo.phybnd_flag[ a ][ i ] )
            {
                m_ph.push_back( pool_phys.at( 2 * a + i ) );
                auto & h = m_ph.back(  );
                h.  side = 2 * a + i;
                h. event = p_dev->event_create    (  );
                h.stream = p_dev->yield_stream    (  );
                h.p_d    = mod.p_data( b.reg ).get(  );
                h.p->n_th_lim = n_th_lim;
                h.p_dev       =    p_dev;
                h.p->setup_p( h, mod );
                assign_strm ( h );
            }
    return;
}

bool keeper_t::desired_mode( const neighbor_t & nb )
{
    return true;
}
 
////////////////////////////////////////////////////////////
// Interface

void keeper_t::update( mesh_t & mesh )
{
    if( ! f_comm_yield )
        throw std::runtime_error( "Unspecified comm_t" );
    
    for( auto & b : mesh )
    {
        assign_comm( b , mesh );
        assign_phys( b , mesh );
    }
    return;
}

};                              // namespace mesh::boundary
