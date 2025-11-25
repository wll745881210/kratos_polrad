#pragma once

#include <map>
#include <mpi.h>

namespace comm
{
////////////////////////////////////////////////////////////
// Device-host buffer pair

using  tag_t = int;
using rank_t = int;

struct buff_t
{
    static       int n_active;
    int              i_active;
    void *                  d;        // Device-side ptr
    void *                  h;        // Host  -side ptr
    size_t                  n;        // size
    size_t                  c;        // capacity
    device::stream_t        s;        // Device stream
    buff_t(   ) : n( 0 ), i_active( 0 ), c( 0 ),
                  d( nullptr ), h( nullptr ) {  };
    
    void  set_active(  )
    {
        i_active = n_active;
    };
    bool test_active(  )
    {
        return ( ( i_active -- ) == n_active );
    };
    void free( device::base_t & f_dev )
    {
        if( h != nullptr )
            f_dev.free_host( h );
    };
};

struct remote_t
{
    ////////// Data //////////
    std::map< tag_t, buff_t > dat;

    ////////// Functions //////////
    buff_t &  operator(  )
    ( void *  p_d, const size_t  & size, const tag_t & tag,
      device::base_t & f_dev, const device::stream_t & str,
      const float & safety = 0.5f )
    {
        auto & b( dat[ tag ] );
        b.d      =   p_d;
        b.s      =   str;
        b.n      =  size;
        b.set_active(  );
        if( size > b.c )
        {
            f_dev.sync_stream        ( b.s );
            b.c = b.n + ( safety   *   b.n );
            b.free                 ( f_dev );
            b.h = f_dev.f_malloc_host( b.c );
            f_dev.sync_stream        ( b.s );
        }
        return b;
    };

    void sync( device::base_t & f_dev )
    {
        for( auto it = dat.begin(  ); it != dat.end(  ); )
        {
            auto  & b( it->second );
            if( b.test_active(  ) )
                f_dev.a_cp( b.d, b.h, b.n, b.s );
            if( b.i_active > 0 )
                ++ it;
            else
            {
                b  .   free( f_dev );
                it = dat.erase( it );
            }
        }
        return;
    };

    void free( device::base_t & f_dev )
    {
        for( auto & [ tag, b ] : dat )
            f_dev.free_host( b.h );
        return;
    };
};

struct local_t
{ 
    ////////// Data //////////
    int                lrank_this;        
    int                     lrank;
    bool                     send;
    bool                   update;
    void *                   dat0; // Head
    size_t               capacity;
    MPI_Win                window;    
    std::map< tag_t, buff_t > dat;

    MPI_Group              p_grup; // Pair-specific group
    MPI_Comm               p_comm; // Pair-specific 

    ////////// Functions //////////
    local_t(  ) : capacity( 0 ), dat0( nullptr ),
                  update  ( false )  {         };
    
    void free( device::base_t & f_dev, const bool & deep )
    {
        if( dat0 != nullptr )
        {
            // if( send )
            //     f_dev.unpin ( dat0 );
            MPI_Win_free( & window );
        }
        if( deep )
        {
            MPI_Group_free( & p_grup );
            MPI_Comm_free ( & p_comm );
        }
        return;
    };    

    void operator (  )
    ( void  * p_d , const size_t & size,
      const tag_t & tag, device::base_t & f_dev,
      const device::stream_t & strm )
    {
        auto &  b =   dat   [ tag ];
        update   |= ( size >= b.n );
        b.d       =  p_d;
        b.n       = size;
        b.s       = strm;
        b.set_active(  );
        return;
    };

    void reset_shared( device::base_t & f_dev )
    {
        free( f_dev, false );
        MPI_Win_allocate_shared
            ( ( send ? capacity : 0 ), 1,
              MPI_INFO_NULL, p_comm, & dat0, & window );
        if( send )
        {
            f_dev.pin( dat0, capacity );
            return;
        }
        MPI_Aint c( 0 );
        int      d( 0 );
        MPI_Win_shared_query
          ( window, lrank_this < lrank, & c, & d, & dat0 );
        if( c != capacity )
            throw std::runtime_error
                ( "MPI_Win inconsistent size.\n" );
        return;
    };
    
    void regularize( device::base_t & f_dev,
                     const float & safety = 0.5f )
    {
        if( ! update )
            return;
        
        size_t shift( 0 );
        for( auto it = dat.begin(  ); it != dat.end(  );  )
        {
            if( it->second.test_active(  ) )
            {
                shift += it->second.n;
                ++ it ;
            }
            else
                it = dat.erase( it );
        }
        if( shift >  capacity )
        {
            capacity = shift + size_t( shift * safety );
            reset_shared( f_dev );
        }
        shift = 0;
        for( auto & [ tag, lbuf ] : dat )
        {
            lbuf.h = ( ( char * )  dat0 ) + shift;
            shift += lbuf.n;
        }
        update = false;
        return;
    };
};

};                                 //    namespace comm
