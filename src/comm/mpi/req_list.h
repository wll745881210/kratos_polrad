#pragma once

#include <list>
#include <mpi.h>
#include <mutex>
#include <vector>

namespace comm
{
////////////////////////////////////////////////////////////
// Lists for MPI_Request
// For multithreading calls of non-blocking MPI calls.

struct req_t
{
    size_t                         i;
    std::vector < MPI_Request  > dat;
    req_t(  ) :   dat( 1024 ), i( 0 ) {  };
    MPI_Request * get(      )
    {
        if( i == dat.size(  ) )
            return nullptr;
        return & dat.at( i ++ );
    };
};  // MPI_I* get the req's addr. and keeps the addr...

struct req_ls_t : public std::list< req_t >
{
    ////////// Data //////////
    std::list< req_t >::iterator i;
    std::mutex                   m;

    ////////// Functions //////////
    req_ls_t( ) : std::list< req_t >( 1 ) { i = begin( ); };

    MPI_Request * get(  )
    {
        const std::lock_guard< std::mutex > lock( m );
        auto * res = i->get (  );
        if( i == end(  ) )
            throw std::runtime_error( "" );
        
        if( res == nullptr )
        {
            ++  i ;
            if( i == end(  )  )
            {
                emplace_back(  );
                i = std::prev( end(  ) );
            }
            res = i->get(  );
        }
        return res;
    };
    void wait(  )
    {
        size_t count( 0 ) ;
        for( auto & r_a : ( * this ) )
        {
            if( r_a.i <= 0 )
                continue;
            std::vector< MPI_Status > stats( r_a.i );
            MPI_Waitall( r_a.i, r_a.dat.data (  ),
                         stats.data(  ) );
            for( const auto & stat : stats )
                if( stat.MPI_ERROR != 0 )
                {
                    int  l(  0  );
                    char e[ 256 ];
                    MPI_Error_string( stat.MPI_ERROR,
                                      e, & l );
                    e[ l ] = '\0';
                    std::cerr << e;                    
                    throw std::runtime_error( e );
                }
            count += r_a.i;
            r_a.i      = 0;
        }
        if( size(  ) > 1 )
        {
            front(  ).dat.resize( 2 * count );
            erase( std::next( begin(  ) ), end(  ) );
        }
        i = begin(  );
        return;
    };
};
};                              // namespace comm
