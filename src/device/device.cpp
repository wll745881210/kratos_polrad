#include <cstdlib>
#include <cstring>
#include <thread>

#include "device.h"

namespace device
{
////////////////////////////////////////////////////////////
// Con-/destructor

__constant__ int * dbg_flag_c ;
static int       * dbg_flag_h ( nullptr );
static int       * dbg_flag_dh( nullptr );

static void handle_dbg( base_t & dev )
{
    ( * dbg_flag_h )  =  0;
    dev.f_mset ( dbg_flag_dh, 0, sizeof( int ), nullptr );
    throw std::runtime_error( "dbg_flag > 0" );
    return;
}

base_t:: base_t(  ) : idx_device( -1 ), num_device( -1 )
{
    f_malloc = [ & ] ( size_t s ) -> void *
               { return std::malloc( s ); };
    f_mset   = [ & ] ( void         * p, const int  & v,
                       const size_t & s, const void * s_ )
               { std::memset( p,  v, s ); };
    f_free   = [ & ] ( void * p )  { std::free( p ); };
    f_cp     = [ & ] ( void * t, const void * s, size_t n )
               { std::memcpy( t, s, n ); };

    f_malloc_host  = f_malloc;
    f_free_host    =   f_free;
    f_cc           =     f_cp;
    max_streams    =     1024;
    idx_streams    =        0;
    head_const     =  nullptr;
    size_const     =        0;
    offset_const   =        0;
    seed_rng       =        1;
    use_dbg        =    false;
    f_malloc_const = [ & ]   ( size_t size ) -> void *
    {
        static const int ali ( 8 );
        const  int s_new( ( size + ali - 1 ) / ali * ali );
        const  int offset = const_offset_get( s_new );
        if( offset + s_new > size_const )
            throw std::runtime_error( "const oversize" );
        void * p = ( ( char * ) head_const ) + offset;
        return p;
    };
    return;
}

////////////////////////////////////////////////////////////
// Initialization

void base_t::init( const input & args, const int & rank )
{
    use_callback  = args.get< bool >
                  ( "device", "use_callback",       true );
    max_streams   = args.get< int  >
                  ( "device", "max_streams", max_streams );
    idx_device    = args.get< int  >
                  ( "device", "idx_device",           -1 );
    seed_rng = args.get< int  >( "device", "seed_rng", 7 );
    use_dbg  = args.get< bool >( "device", "use_dbg",  0 );
    num_device = 0;
    return;
}

void base_t::prepare(  )
{
    if( dbg_flag_h == nullptr )
    {
        dbg_flag_dh = malloc_device< int > (  );
        dbg_flag_h  = malloc_host  < int > (  );
        cp_const( dbg_flag_c,  dbg_flag_dh  );
        f_mset( dbg_flag_dh, 0, sizeof( int ), nullptr );
        ( * dbg_flag_h )   = 0;
    }
    return;
}

void base_t::finalize(  )
{
    if( dbg_flag_h != nullptr )
    {
        free_device( dbg_flag_dh );
        free_host  ( dbg_flag_h  );
    }
    return;
}

////////////////////////////////////////////////////////////
// Callback

std::vector< std::function< void(  ) > > base_t::cb_vec;

void base_t::callback_std( void * p )
{
    return cb_vec.at( ( intptr_t )( p ) )(  );
}

void base_t::launch_host( const stream_t & s, func_t && f )
{
    if( ! use_callback )
    {
        sync_stream( s );
        return    f(   );
    }
    cb_vec.push_back( f );
    void * idx = reinterpret_cast
               < void * >( cb_vec.size(  )  - 1 );
    return f_launch_host ( s, callback_std, idx );
}

////////////////////////////////////////////////////////////
// Stream and events

void base_t::sync_stream( const stream_t & stream )
{
    if( use_dbg )
        cp_a( dbg_flag_h, dbg_flag_dh, 1, stream );
    if( !  stream_m.empty(  ) )
    {
        auto it  =  stream_m.find( stream );
        if(  it !=  stream_m. end(      ) )
        {
            for( auto & event : it->second )
                event_sync( event );
            it->second.clear(     );
        }
    }
    if( ! sstream_m.empty(  ) )
    {
        auto it  = sstream_m.find( stream );
        if(  it != sstream_m. end(      ) )
        {
            for( auto & stream : it->second )
                sync_stream_base   ( stream );
            it->second.clear(  );
        }
    }
    sync_stream_base ( stream );
    if( use_dbg && ( * dbg_flag_h ) > 0 )
        handle_dbg ( * this );
    return;
}

void base_t::sync_all_streams(  )
{
    for( auto & [ s, e ] : stream_m )
    {
        for( auto & event  : e )
             event_sync( event );
        e.clear(  );
    }
    for( auto & [ e, s ] :  event_m )
        sync_stream_base ( s );
    event_m.clear(   );
    
#ifdef __GPU_DEBUG__
    cp( dbg_flag_h, dbg_flag_dh );
    if( * dbg_flag_h > 0 )
        handle_dbg( * this );
#endif
    return;
}

void base_t::stream_record( const stream_t & stream ,
                            const event_t  &  event )
{
    event_m[ event ] = stream ;
    return   stream_m[ stream ].push_back  (  event );
}

void base_t::stream_record( const stream_t & str_w ,
                            const stream_t & str_s )
{
    if( str_w != str_s )
        sstream_m [ str_w ]. push_back ( str_s );
    return;
}

////////////////////////////////////////////////////////////
// Constant memory pool

int base_t::const_offset_get( const int & add )
{
    offset_const += add ;
    return offset_const - add;
}

};                              // namespace device
