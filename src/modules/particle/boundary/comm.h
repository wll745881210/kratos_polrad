#pragma once

#include "../../../utilities/types/crtp.h"
#include "../types/driver_base.h"
#include "../types/particle_pool.h"

#include <set>
#include <vector>

namespace particle::comm
{
////////////////////////////////////////////////////////////
// Device-side kernel interface

template < class com_T , class pol_T >  __global__
void ker_cpy( const com_T comm, const pol_T pool )
{
    return comm.cpy_d( pool );
};

////////////////////////////////////////////////////////////
// Communication handler

template< class pol_T, class derived_T = crtp::dummy_t >
struct comm_t : driver::base_t
{
    ////////// Type //////////
    __crtp_def_self__( comm_t, derived_T, pol_T );
    
    struct reg_t
    {
        size_t        i_par;
        int           i_seq;
    };

    ////////// Map to rank //////////
    int              n_rank;
    int    *           rank;

    ////////// Data //////////
    pol_T  *           send;
    pol_T  *         send_h;
    int    *         n_send;
    int    *       n_send_t;
    int    *       n_send_h;
    pool_t < reg_t > cp_idx;

    __host__ comm_t(  ) : n_rank( 0 ), rank( nullptr ) {  };

    __host__ virtual void set_mem( device::base_t & f_dev )
    {
        send     = f_dev.malloc_device< pol_T  > ( n_rank );
        rank     = f_dev.malloc_host  < int    > ( n_rank );
        n_send_h = f_dev.malloc_host  < int    > ( n_rank );
        n_send   = f_dev.malloc_device< int >( 1 + n_rank );
        n_send_t = n_send  +  n_rank  ;

        send_h   = new pol_T[ n_rank ];
        f_dev.pin( send_h, sizeof( pol_T ) * n_rank );
        return;
    };

    __host__ virtual void update
    ( const mesh::mod_base_t::v_reg_t & news,
      particle::               base_t & mod )
    {
        const auto rank_this( mod.p_bdk->p_com->rank(  ) );
        std::set < int  >    ranks ;
        for( const auto & b : ( * mod.p_mesh ) )
            for( const auto & [ nb, reg ] : b.neighbors )
                if( nb.guest_info.rank  !=    rank_this )
                    ranks.insert  ( nb. guest_info.rank );

        if( ranks.size(  ) >  n_rank )
        {
            n_rank   =   ranks.size(   );
            this->finalize ( mod );
            this->set_mem( * mod.p_dev );
        }
        int  i_rank   ( 0 );
        for( const auto & r_s : ranks )
        {
            this->rank[ i_rank ] =  r_s;
            ++          i_rank ;
        }
        return mod.p_dev->sync_stream( mod.stream );
    };
    __host__ void finalize( particle::base_t & mod )
    {
        if( rank == nullptr )
            return;
        for( int i = 0; i < n_rank; ++  i )
            send_h[ i ].finalize    ( mod );
        mod.p_dev->free_device( n_send    );
        mod.p_dev->free_device(   send    );
        mod.p_dev->free_host  (   rank    );
        mod.p_dev->free_host  ( n_send_h  );
        mod.p_dev->unpin      (   send_h  );
        delete     [  ] send_h;
        return;
    };
    __host__ __forceinline__ bool skip_comm(  )
    {
        return ( n_rank == 0 );
    };

    __host__ void prep
    ( const pol_T & pool, particle::base_t & mod )
    {
        if( skip_comm(  ) )
            return;  // v-- Set n_send and n_send_t to 0
        mod.p_dev->f_mset( n_send,   0, ( 1 + n_rank ) *
                           sizeof( int ), mod.stream ) ;
        cp_idx.set_mem( mod, pool.n_par, 1.5 ) ;
        return;
    };

    __device__ __forceinline__ void reg
    ( const size_t & i_par, const int & i_rank ) const
    {
        auto & ci = cp_idx[ atomicAdd( const_cast< int * >
                                     ( n_send_t ), 1 ) ] ;
        ci.i_par  = i_par;
        ci.i_seq  = atomicAdd( const_cast <  int * >
                             ( n_send ) + i_rank ,   1 ) ;
        return;
    };

    __host__ void sync_proc( device  ::base_t & dev ,
                             particle::base_t & mod )
    {
        // return dev.sync_all_streams(  );
        return dev.sync_stream( mod.stream );
    };
    
    __host__  bool  proc
    ( pol_T & pool, particle::base_t & mod )
    {
        auto & f_com( * mod.p_bdk -> p_com );
        f_com.barrier(  );
        if( skip_comm(  ) )
        {
            pool.regulate( mod    );
            pool.set_recv( mod, 0 );
            return false;
        }
        size_t s_i( sizeof( int ) );
        auto & f_dev( * mod.p_dev );
        f_dev.cp_a( n_send_h, n_send, n_rank, mod.stream );
        sync_proc( f_dev, mod );

        std::vector< int > n_recv_h( n_rank );
        int n_comm_global( 0 ) ;
        for( int i = 0; i < n_rank;  ++ i )
        {
            if( f_com.rank(  ) == rank[ i ] )
                continue;
            f_com.isend_h( & n_send_h [ i ], s_i,
                           rank[ i ], f_com.rank(   ) );
            f_com.irecv_h( & n_recv_h[ i ], s_i,
                           rank[ i ],       rank[ i ] );
        }
        f_com. wait_all_h(   );
        int n_recv       ( 0 );
        for( int i = 0; i < n_rank; ++ i )
            n_recv   += n_recv_h[ i ];
        n_comm_global = n_recv;
        f_com.reduce_all( n_comm_global, ::comm::sum );
        sync_proc ( f_dev, mod );        
        bool reset_send( false );
        for( int i  = 0; i < n_rank; ++ i )
            reset_send |=  send_h[ i ].set_mem
                       ( mod, n_send_h[ i ], 2. );
        sync_proc ( f_dev, mod );
        if( reset_send )
            f_dev.cp_a( send, send_h, n_rank, mod.stream );
        copy_buf( pool,   mod );
        pool.regulate   ( mod );
        sync_proc( f_dev, mod );
        pool.set_recv ( mod, n_recv );
        if( n_comm_global < 1 )
        {
            sync_proc( f_dev, mod );
            return false;
        }
        for( int i = 0, s_recv = 0; i < n_rank; ++ i )
        {
            if( f_com.rank(  ) == rank[ i ] )
                continue;
            if( send_h[ i ].n_par > 0 )
                f_com.isend_d
                ( & send_h[ i ][ 0 ], send_h[ i ].size(  ),
                  rank[ i ], f_com.rank(  ),  mod.stream );
            if( n_recv_h[ i ] > 0 )
                f_com.irecv_d
                    ( pool.pdata( s_recv ), n_recv_h[ i ] *
                      pool.unit_size (   ), rank[ i ],
                      rank[ i ],  mod.stream );
            s_recv += n_recv_h[ i ];
        }
        f_com.wait_all_d(  );
        sync_proc( f_dev, mod );
        if( mod.comm_mode(  ) == refresh_pool )
            return ( n_comm_global > 0 );
        else
            return false;
    };

    __host__ void copy_buf
    ( pol_T & pool, particle::base_t & mod ) const
    {
        int  n_s_t ( 0 );
        for( int i = 0; i < n_rank; ++ i )
            n_s_t += n_send_h[ i ];
        if( n_s_t > 0 )
        {
            const auto n_bl( ( n_s_t + n_th - 1 ) / n_th );
            const auto lpar = std::make_tuple
                ( dim3( n_bl ), dim3( n_th ), 0 );    
            mod.p_dev->launch( ker_cpy, lpar, mod.stream,
                               get_self (  ), pool );
        }
        return;
    };

    __device__ __forceinline__
    void cpy_d( const pol_T & pool ) const
    {
        const auto i = utils::th_id(  );
        if( i >= ( * n_send_t ) )
            return;
        const auto & ci  = cp_idx[ i ];
        auto & src = pool[ ci. i_par ];
        const  int i_rank( src.dest.   i_rank );
        auto & tgt = send[ i_rank ][ ci.i_seq ];
        src.move( tgt );
        tgt.dest.todo = to_keep;
        return;
    };  // Do NOT set the src.dest, otherwise not deleted
};
};                          // namespace particle::comm
