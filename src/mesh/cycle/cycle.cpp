#include "cycle.h"

#include <cfloat>
#include <csignal>
#include <iomanip>
#include <thread>

#include "../../utilities/mapping/loop.h"
#include "../../utilities/types/my_type_traits.h"
#include "../mesh.h"

namespace mesh::cycle
{
////////////////////////////////////////////////////////////
// "Local" quantities

using type::float_t;

////////////////////////////////////////////////////////////
// Con-/destructor and initializer

base_t:: base_t(  ) : i_output( 0 ), t_lim   ( FLT_MAX ),
    i_cycle( 0 ), n_cycle_lim ( 0 ), n_display_freq( 1 ),
    t( 0 ),       dt_expand   ( 2 ), t_display_prec( 6 ),
    dt_reduced( false )
{
    return;
}

void base_t::init( const input & args, mesh_t & mesh )
{
    signal( SIGINT,  sighdl );
    //////////////////////////////////////////////////
    // Evolution control
    p_dev   = mesh . p_dev;
    stream  = p_dev->yield_stream       (           );
    p_dt    = p_dev->malloc < float_t > ( 1,  false );
    p_dt_h  = p_dev->malloc < float_t > ( 2,   true );
    p_dt_h[ 0 ] =  0;
    args( "cycle", "t_0",                         t );
    args( "cycle", "t_lim",                   t_lim );
    args( "cycle", "dt_init",              * p_dt_h );
    args( "cycle", "dt_expand",           dt_expand );
    args( "cycle", "n_cycle_lim",       n_cycle_lim );

    //////////////////////////////////////////////////
    // Information output (on-screen display)
    args( "cycle", "t_display_prec", t_display_prec );
    args( "cycle", "n_display_freq", n_display_freq );

    prefix       = args.get< std::string >
                 ( "cycle", "prefix_output", "test" );
    t_output     = args.get< float2_t >
                 ( "cycle",  "t_output_next",  0.00 );
    dt_output    = args.get< float2_t >
                 ( "cycle", "dt_output",       1e32 );
    final_output = args.get< bool >
                 ( "cycle",  "final_output",  false );
    if( mesh.is_restart(  ) )
        load ( * mesh.p_bio );
    return;
}

void base_t::finalize(  )
{
    p_dev->free( p_dt,  false );
    p_dev->free( p_dt_h, true );
    return;
}

////////////////////////////////////////////////////////////
// Evolution

void base_t::redc_dt_start( mesh_t & mesh )
{
    if( dt_reduced )
        return;
    p_dev->cp_a( p_dt_h, p_dt, 1, stream );
    p_dev->sync_stream          ( stream );
    mesh.p_com->reduce_all( * p_dt_h, comm::min, true );
    return;
}

void base_t::redc_dt_finish( mesh_t & mesh )
{
    if( dt_reduced )
        return;
    mesh.p_com->reduce_all_finish(  );
    p_dt_h[ 0 ] =  utils::min( * p_dt_h, t_lim - t );
    t           =  utils::min( * p_dt_h + t, t_lim );
    p_dt_h[ 1 ] =  ( * p_dt_h ) *  dt_expand  ;
    p_dev->cp_a( p_dt, p_dt_h + 1, 1, stream );
    p_dev->sync_stream             (  stream );
    dt_reduced = true;
    return;
}

float_t base_t::dt(  ) const
{
    return p_dt_h[ 0 ] ;
}

bool base_t::step( mesh_t & mesh )
{
    struct
    {
        bool       to_save;
        float2_t  t_output;
        float2_t t_current;
    }   t_sync;
    t_sync.to_save   = update_tout( t_output, dt_output );
    t_sync.t_current = t;
    t_sync.t_output  = t_output;
    mesh.p_com->bcast( & t_sync, sizeof( t_sync ) );
    t        = t_sync.t_current;
    t_output = t_sync. t_output;
    
    if( t_sync.to_save
        && ! std::isnan( t_output + dt_output ) &&
           ! std::isinf( t_output + dt_output )  )
        mesh.save( file_name( mesh ) );
    if( t   >= t_lim )
        return false ;
    mesh.step(  ) ;
    if( i_cycle % n_display_freq == 0 )
        print_info( mesh );
    dt_reduced = false;
    return        true;
}

void base_t::evolve( mesh_t & mesh )
{
    i_cycle_0  = i_cycle;
    n_cell_tot = 0.;
    for( auto & b : mesh )
        n_cell_tot += b.geo.n_ceff.prod(  );
    mesh.p_com->reduce_all( n_cell_tot, comm::sum );

    wt_prev    = clock::now(  );
    auto start = wt_prev;

    if( p_dt_h[ 0 ] < FLT_MIN )
        p_dt_h[ 0 ] = FLT_MAX / 10 ;
    p_dt_h[ 1 ]  /= dt_expand ;
    mesh.p_dev->cp( p_dt, p_dt_h, 1 );
    mesh.p_dev->sync_all_streams(   );
    bool to_stop( false );
    try
    {
        while( step( mesh ) && i_cycle < n_cycle_lim )
            ++ i_cycle;
    }
    catch( const std::exception & exc )
    {
        std::cerr << exc.what(  );
        to_stop = true;
    }
    if( to_stop && final_output )
        return mesh.save( file_name( mesh ) );
    
    mesh.p_dev->sync_all_streams(  );
    mesh.p_com->barrier         (  );
    const auto    dwt   = dt_wt( clock::now(  ) - start );
    const float_t speed = n_cell_tot / dwt
                        * (  i_cycle - i_cycle_0  +   1 );
    if( mesh.p_com->is_root(  ) )
        std::cout << "Duration = " << dwt   << " s, "
                  << "Speed = "    << speed << " cell/s,\n"
                  << "Pace = " << 1 / speed << " s/cell.\n";
    if( final_output && ! std::isnan( t_output ) )
        mesh.save( file_name( mesh ) );
    return;
}

////////////////////////////////////////////////////////////
// Signals

void base_t::sighdl( int signum )
{
    std::stringstream ss;
    ss << "\nSignal : " << signum  << '\n';
    throw std::runtime_error( ss.str(  ) );
    return;
};

////////////////////////////////////////////////////////////
// Speed tests

template< class c_T >
float_t base_t::dt_wt( const c_T & dt_clk )
{
    std::chrono::duration< float_t > dt( dt_clk );
    return dt.count(  );
}

float_t base_t::dt_wt(  )
{
    const auto now = clock::now(  );
    const auto res = dt_wt( now - wt_prev );
    wt_prev  = now;
    return     res;
}

////////////////////////////////////////////////////////////
// Information output to stdout

bool base_t::update_tout
( float2_t & tout, const float2_t & dt )
{
    const auto delta_t = t - tout;
    if( delta_t < 0 )
        return false;
    const size_t  n_dt = size_t( delta_t / dt ) + 1;
    tout  +=      n_dt * dt;
    return true;
}

void base_t::print_info( mesh_t & mesh )
{
    if( ! mesh.p_com->is_root(  ) )
        return;
    const auto speed = n_cell_tot / dt_wt(  )
        * std::min( n_display_freq, i_cycle - i_cycle_0 );
    std::cout << "cycle = " << std::setfill( ' ' )
              << std::setw( 5 )<< i_cycle << std::scientific
              << std::setprecision(  t_display_prec )
              << ", t = " << t << ", dt = " << dt(  )
              << ", Speed = "  << speed << std::endl;
    return;
}

////////////////////////////////////////////////////////////
// Binary IO

std::string base_t::file_name( mesh_t & mesh )
{
    std::string         res;
    if( mesh.p_com->is_root(  ) )
    {
        std::stringstream ss;
        ss << prefix << "_" << std::setfill( '0' )
           << std::setw( 5 ) << i_output << ".bin";
        std::cout << "Making output at cycle = " << i_cycle
                  << ", t = " << t << "; file name: "
                  << ss.str(  ) << std::endl;
        res = ss.str(  );
    }
    mesh.p_com->bcast( res );
    return res;
}

void base_t::save( binary_io::base_t & bio )
{
    ++  i_output;
    bio.write( t,         "time" );
    bio.write( p_dt_h,   1, "dt" );
    bio.write( i_cycle,  "cycle" );
    bio.write( i_output, "i_out" );
    bio.write( t_output, "t_out" );
    return;
}

void base_t::load( binary_io::base_t & bio )
{
    bio.read ( t,         "time" );
    bio.read ( p_dt_h,      "dt" );
    bio.read ( i_cycle,  "cycle" );
    bio.read ( i_output, "i_out" );
    bio.read ( t_output, "t_out" );
    return;
}

};                              // namespace mesh::cycle
