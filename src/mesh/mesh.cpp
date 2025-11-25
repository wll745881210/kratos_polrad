#include "mesh_enroll.h"
#include <cfenv>

namespace mesh
{
////////////////////////////////////////////////////////////
// Constructor, destructor

mesh_t:: mesh_t(  ) : order_max( 0 ), restart( false )
{
    return;
}

mesh_t::~mesh_t(  )
{
    for( auto  & p :   mods )
        p->finalize( * this );
    if( p_cyc )
        p_cyc->finalize(    );
    if( p_bio )
        p_bio->finalize(    );
    if( p_com )
        p_com->finalize(    );    
    if( p_dev )
        p_dev->finalize(    );
    return;
}

////////////////////////////////////////////////////////////
// Initialization

void mesh_t::init( const input & args )
{
    //////////////////////////////////////////////////
    // Error on float exception to ease debugging
    if( args.get< bool >( "cycle", "except_ferr", false ) )
        feenableexcept
            ( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );
    
    //////////////////////////////////////////////////
    // Computing modules' initializaitons come first
    if( ! p_dev )
        throw std::runtime_error( "device unspecified" );
    if( ! p_bio )
        enroll_binary_io< binary_io::default_t >    (  );
    if( ! p_mgn )
        enroll_mgen     < meshgen  ::   base_t >    (  );
    if( ! p_cyc )
        enroll_cycle    < cycle    ::   base_t >    (  );
    if( ! p_com )
        enroll_comm     < comm     ::   base_t >    (  );
    if( ! p_tre )
        enroll_tree     <               tree_t >    (  );
    if( ! f_blk_yield )
        enroll_block    <              block_t >    (  );

    //////////////////////////////////////////////////
    // Initialization
    for( auto & p : reads )
        p         (  args );

    input_args = args.get_item_map   (        );
    restart    = args.found( "file", "binary" );
    if( restart )
    {
        p_bio->open( args.get  < std::string >
                   ( "file", "binary" ), "r" );
        p_bio->load(                  );
    }
    p_com->init   ( args );
    p_dev->init   ( args, p_com->l_rank(   ) );
    p_cyc->init   ( args, * this );
    p_mgn->init   ( args, * this );
    for( auto & p : inits )
        p.second  (  args );
    if( restart )
        p_bio->close(     );
    return;
}

const bool & mesh_t::is_restart(  ) const
{
    return restart;
}

bool mesh_t::changed(  ) const
{
    return false;
}

const int & mesh_t::n_order(  ) const
{
    return order_max;
}

////////////////////////////////////////////////////////////
// Binary IO

void mesh_t::save( const std::string & file_name )
{
    p_bio->open ( file_name,           "w" );
    p_bio->write( input_args, "input_args" );
    if( p_com->is_root (  ) )
        p_cyc->save    (  * p_bio );
    p_mgn->save( * this,  * p_bio );
    p_dev->sync_all_streams (     );
    for( auto & p : mods )
        p->save ( * this, * p_bio );
    p_bio->save (  );
    p_bio->close(  );
    return;
}

////////////////////////////////////////////////////////////
// Mapping-based block interface for the base class

tree_t & mesh_t::tree(  ) const
{
    return const_cast< tree_t & > ( * p_tre );
}

block_t & mesh_t::emplace( const region_logic_t & reg )
{
    if( contains( reg ) )
        throw std::runtime_error( "mesh_t::emplace" ) ;

    auto p = f_blk_yield(  );
    block_map[ reg ] = p;
    return ( * p );
}

bool mesh_t::contains( const region_logic_t & reg )
{
    return block_map.find( reg ) != block_map.end(  );
}

void mesh_t::remove  ( const region_logic_t & reg )
{
    // #warning "Change here after implementing AMR"
    // tree.erase( reg );
    block_map.erase( reg );
    return;
}

block_t & mesh_t::block( const region_logic_t & reg )
{
    auto r = reg;
    tree(  ).regularize( r );
    return ( * block_map.at ( r ) );
}

block_t & mesh_t::front(   )
{
    return ( * block_map. begin(  )->second );
}

block_t & mesh_t:: back(   )
{
    return ( * block_map.rbegin(  )->second );
}

////////////////////////////////////////////////////////////
// Range-based for loop

mesh_t:: iter_t::iter_t
( const  mesh_t::block_map_t::iterator & it )
{
    static_cast< block_map_t::iterator & >( * this ) = it;
}

block_t & mesh_t::iter_t::operator * (  )
{
    return ( * static_cast< block_map_t::iterator & >
           ( * this )->second );
}

mesh_t::iter_t mesh_t::begin(  )
{
    return   block_map.begin(  );
}

mesh_t::iter_t mesh_t::  end(  )
{
    return   block_map.  end(  );
}

////////////////////////////////////////////////////////////
// Evolution

void mesh_t::step  (  )
{
    p_mgn  ->update   ( *  this );
    for(  auto & p_step : steps )
        p_step.second(  );
    restart   = false;  // No longer restart after one step
    return;
}

void mesh_t::evolve(  )
{
    p_cyc  ->evolve ( * this );
    return p_com->finalize(  );
}

};                              // namespace mesh
