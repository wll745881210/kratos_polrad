#ifdef  __MPI__                  // Skip if not MPI-enabled

#include "mpi.h"
#include <thread>

namespace comm
{
////////////////////////////////////////////////////////////
// Type

int buff_t::n_active( 100 );

////////////////////////////////////////////////////////////
// Constructor

int mpi_t::count( 0 );          // Total MPI instances

mpi_t:: mpi_t(  )
{
    int ini( 0 ),    mt( 0 );
    MPI_Initialized( & ini );
    if( ini == 0 )
    {
        MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, & mt );
        if( mt < MPI_THREAD_MULTIPLE )
            throw std::runtime_error( "Not multi-th MPI" );
    }
    root = 0;
    ++ count;
    MPI_Comm_dup ( MPI_COMM_WORLD, & comm );
    MPI_Comm_rank( comm, &   rank_ );
    MPI_Comm_size( comm, & n_rank_ );
    local_init   (                 );
    local_optimize           =  true;
    wait_background          = false;
    return;
}

mpi_t::~mpi_t(  )
{
    local_finalize (  );
    remote_finalize(  );
    if( comm != MPI_COMM_WORLD )
        MPI_Comm_free( & comm );
    if( ( -- count ) == 0 )
    {
        int flag      ( 0      );
        MPI_Finalized ( & flag );
        if( flag == 0 )
            MPI_Finalize(  );
    }
    return;    
}

void mpi_t::init( const input & args )
{
    local_optimize   = args.get < bool >
        ( "comm", "local_optimize",  1 );
    wait_background  = args.get < bool >
        ( "comm", "wait_background", 0 );
    buff_t::n_active = args.get <  int >
        ( "comm", "buff_n_active", 100 );
    return;
}

std::shared_ptr< base_t > mpi_t::split(  ) const
{
    auto  res = std::make_shared  < mpi_t > (  );
    res-> local_optimize = this-> local_optimize;
    res->wait_background = this->wait_background;
    return res;
}

////////////////////////////////////////////////////////////
// Direct memory access for node-local processes

template< class fun_T >
void mpi_t::map_local( const fun_T & f )
{
    for( int m = 0; m < nl_rank; ++ m )
        if( m != l_rank_ )
            f( m, m < l_rank_ ? local_send : local_recv ,
                  m < l_rank_ ? local_recv : local_send );
    return;
}

bool mpi_t::is_local( const int & rank )
{
    if( local_optimize )
        return rank >= rank_l_lim[ 0 ]
            && rank <= rank_l_lim[ 1 ];
    return false;
}

void mpi_t::local_init(  )
{
    MPI_Comm_split_type( comm , MPI_COMM_TYPE_SHARED,
                         rank_, MPI_INFO_NULL, & l_comm );
    MPI_Comm_rank ( l_comm, &    l_rank_ );
    MPI_Comm_size ( l_comm, &   nl_rank  );

    local_send.resize( nl_rank );
    local_recv.resize( nl_rank );

    rank_l_lim[ 0 ] = -rank_;
    rank_l_lim[ 1 ] =  rank_; // Local limit of ranks
    MPI_Allreduce( MPI_IN_PLACE, rank_l_lim, 2,
                   MPI_INT, MPI_MAX,  l_comm );
    rank_l_lim[ 0 ] *= -1;

    MPI_Group                 l_group  ;
    MPI_Comm_group( l_comm, & l_group );
    for( int m = 0 ; m < nl_rank - 1; ++ m )
        for( int n = m + 1 ; n < nl_rank; ++ n )
        {
            MPI_Group g;
            MPI_Comm  c;
            int l_ranks [ 2 ] = { m, n };
            MPI_Group_incl  ( l_group, 2, l_ranks, & g );
            MPI_Comm_create ( l_comm , g, & c );
            if( m == l_rank_ || n == l_rank_  )
            {
                const int lrank_tgt( m == l_rank_? n : m );
                auto & lb_s( local_send[ lrank_tgt ] ) ;
                auto & lb_r( local_recv[ lrank_tgt ] ) ;
                for( auto * p : { & lb_s, & lb_r } )
                {
                    p->    p_grup =         g;
                    p->    p_comm =         c;
                    p->lrank_this =   l_rank_;
                    p->lrank      = lrank_tgt;
                }
                lb_s.send =  true;
                lb_r.send = false;
            }
            MPI_Barrier( l_comm );
        }    
    return;
}

void mpi_t::local_finalize(  )
{
    MPI_Barrier( l_comm );
    map_local( [ & ] ( const int & m, auto & l0, auto & l1 )
    {
        l0[ m ].free ( * p_dev, false );
        l1[ m ].free ( * p_dev,  true );        
    }   );
    return;
}

void mpi_t::local_synchronize(  )
{
    if( ( ! local_optimize ) || nl_rank <= 1 )
        return this->barrier(  );

    map_local( [ & ] ( const int & m, auto & l0, auto & l1 )
    {
        l0[ m ].regularize( * p_dev );
        l1[ m ].regularize( * p_dev );        
    }  );
    for( auto & local : local_send )
        for( auto & [ rag,  b ] : local. dat )
        {
             p_dev->a_cp( b.h, b.d, b.n, b.s );
             p_dev->sync_stream        ( b.s );
        }
    MPI_Barrier( l_comm );
    for( auto & local : local_recv )
        for( auto & [ rag,  b ] : local. dat )
             p_dev->a_cp( b.d, b.h, b.n, b.s );            
    return;
}

////////////////////////////////////////////////////////////
// Conventional send and recv

void mpi_t::remote_finalize(  )
{
    for( auto * p_b : { & buf_send, & buf_recv } )
        for( auto & [ tag, b ] : ( * p_b ) )
            b.free( *  p_dev ) ;
    return;
}

void mpi_t::remote_synchronize(  )
{
    if( buf_send.size(  ) < 1 && buf_recv.size(  ) < 1 )
        return;
    for( auto & [ r,  buf ] : buf_send )
        buf.sync( * p_dev ) ;        
    reqs_d.wait(  );
    for( auto & [ r , buf ] : buf_recv )
        buf.sync( * p_dev ) ;
    return;
}

////////////////////////////////////////////////////////////
// Deal with requests

void mpi_t::wait_all_d(  )
{
    if( wait_background )
    {
        std::thread th( [ & ] (  )
        {
            remote_synchronize(  );
        }   );
        local_synchronize     (  );
        return th.join        (  );
    }
    remote_synchronize        (  );
    return local_synchronize  (  );
}

void mpi_t::wait_all_h(  )
{
    reqs_h.wait  (  );
    this->barrier(  );    
}

////////////////////////////////////////////////////////////
// Device-side aysnc comm

void mpi_t::isend_d
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag ,
  const device:: stream_t      & stream )
{
    if( this->is_local( rank ) )            
        local_send[ rank - rank_l_lim[ 0 ] ]
            ( data, size, tag, * p_dev, stream );
    else
    {
        auto * pb  = & buf_send[ rank ]
                     ( data, size, tag, * p_dev, stream );
        p_dev ->a_cp ( pb->h, pb->d, pb->n, stream );
        p_dev->launch_host ( stream, [ = ] (  )           
        {
            auto * pr = this->reqs_d.get(  );
            MPI_Isend( pb->h, pb->n, MPI_CHAR, rank,
                       tag, comm, pr );
            if( ( * pr ) == 0 )
                throw std::runtime_error( "" );            
        }   );
    }
    return;
}

void mpi_t::irecv_d
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag ,
  const device:: stream_t      & stream )
{
    if( this->is_local( rank ) )
        local_recv[ rank - rank_l_lim[ 0 ] ]
            ( data, size, tag, * p_dev, stream );
    else
    {
        auto & b = buf_recv[ rank ]
                     ( data, size, tag, * p_dev, stream );
        auto * pr = reqs_d.get(  ) ;
        
        MPI_Irecv( b.h, b.n, MPI_CHAR, rank,
                   tag, comm, pr );
        if( ( * pr ) == 0 )
            throw std::runtime_error( "" );
    }
    return;
}

////////////////////////////////////////////////////////////
// Host-side async comm

void mpi_t::isend_h
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag )
{
    auto * pr = reqs_h.get(  ) ;    
    MPI_Isend( data, size, MPI_CHAR, rank, tag, comm, pr );
    if( ( * pr ) == 0 )
        throw std::runtime_error( "" );    
    return;
}

void mpi_t::irecv_h
( void      * data, const size_t & size ,
  const int & rank, const int    &  tag )
{
    auto * pr = reqs_h.get(  ) ;        
    MPI_Irecv( data, size, MPI_CHAR, rank, tag, comm, pr );
    if( ( * pr ) == 0 )
        throw std::runtime_error( "" );        
    return;
}

////////////////////////////////////////////////////////////
// Host-side broadcast

void mpi_t::bcast( void * p, const size_t & size )
{
    if( n_rank(  ) > 1 )
        MPI_Bcast( p, size, MPI_CHAR, root, comm );
    return;
}

void mpi_t::barrier(  )
{
    MPI_Barrier( comm );
    return;
}

////////////////////////////////////////////////////////////
// Host-side reduction

void mpi_t::reduce_all_ker
( void  * p,  const std::type_info & t ,
  const operation_t & o, const int & n,
  const bool & async )
{
    if( n_rank(  ) <= 1 )
        return;
    MPI_Op       op( MPI_OP_NULL       );
    MPI_Datatype tp( MPI_DATATYPE_NULL );

    switch( o )
    {
    case min:
        op = MPI_MIN;
        break;
    case max:
        op = MPI_MAX;
        break;
    case sum:
        op = MPI_SUM;
        break;
    default:
        break;
    };
    if     ( t == typeid(    int ) )
        tp = MPI_INT;
    else if( t == typeid(  float ) )
        tp = MPI_FLOAT;
    else if( t == typeid( double ) )
        tp = MPI_DOUBLE;
    else if( t == typeid( size_t ) )
        tp = MPI_UNSIGNED_LONG;
    else
        throw std::runtime_error( "Undefined reduce type" );
    
    if( async )
        MPI_Iallreduce( MPI_IN_PLACE, p, n, tp, op, comm,
                        & reduce_req );
    else
        MPI_Allreduce( MPI_IN_PLACE, p, n, tp, op, comm );
    return;
}

void mpi_t::reduce_all_finish(  )
{
    MPI_Wait( & reduce_req, MPI_STATUSES_IGNORE );
    return;
}

};                              // namespace communicate
#endif // __MPI__
