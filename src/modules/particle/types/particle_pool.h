#pragma once

#include "../../../utilities/mapping/reduction.h"
#include "driver_base.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Container for particles

template< class pol_T, class int_T >
__global__ void ker_pool_relocate
( const pol_T pool, const int_T n_keep, const int_T n_mv )
{
    return pool.relocate( n_keep, n_mv );
};

template  < class     par_T , class int_T = int,
            class derived_T = crtp::dummy_t >
struct pool_t : driver::base_t
{
    ////////// Type //////////
    __crtp_def_self__( pool_t, derived_T, par_T, int_T );
    using par_t     =   par_T;
    using int_t     =   int_T;
    comm_mode_t     comm_mode;

    ////////// Data //////////
protected:                  // Intentional, avoid misuse
    char  *            par;
public:
    bool            output;
    int_T            n_par;
    int_T           n_recv;
    int_T           offset;
    int_T         capacity;
    int_T *       map_pars;
    int_T *         n_mv_h;
    int_T *           i_mv;

    ////////// Initialize/finalize //////////
    __host__ pool_t(   )
        : par( nullptr ), i_mv( nullptr ), n_par( 0 ),
          capacity ( 0 ), n_recv( 0 ),    offset( 0 ),
          map_pars( nullptr )   {   };

    __host__ virtual void init
    ( const  input & args, particle::base_t & mod ) override
    {
        output = args.get< bool >
                 ( "particles", "output", true );
        return driver::base_t::init( args, mod );
    };

    __host__ virtual void finalize( particle::base_t & mod )
    {
        if( par != nullptr )
        {
            mod.p_dev->free_host  (   n_mv_h );
            mod.p_dev->free_device( map_pars );
            mod.p_dev->free_device(     i_mv );
            mod.p_dev->free_device(      par );
        }
        return;
    };

    __host__  void read ( const mesh:: f_read_t & ) {  };
    __host__  void write( const mesh::   f_cp_t & ,
                          const mesh::f_write_t & ) {  };

    __host__ __device__ __forceinline__
    size_t unit_size(  ) const
    {
        return sizeof( par_T );
    };

    __host__ virtual  bool set_mem
    ( particle::base_t & mod, const int_T & n_par,
      const type ::float2_t & safe = 1.1,
      const bool & keep_data = true )
    {
        comm_mode = mod.comm_mode(  );
        auto & dev( * mod.p_dev );
        if( i_mv == nullptr )
        {
            n_mv_h  = dev.malloc_host      < int_T >( 2 );
            i_mv    = dev.malloc_device    < int_T >( 2 );
            dev.f_mset( i_mv, 0, 2 * sizeof( int_T ), 0 );
        }
        if( capacity >= n_par )
        {
            this-> n_par = n_par ;
            return false ;
        }
        capacity = n_par + n_par * ( safe - 1 );

        dev.sync_stream( mod.stream );
        char * par = dev.malloc_device< char >
             ( capacity * get_self(  ).unit_size(  ) );
// #warning "TODO: Blocked pool for size increase!"
        if( this->par != nullptr )
        {
            if( keep_data )
                dev.f_cp( par, this->par, this->size(  ) );
            dev.free_device( this->par );
            dev.free_device(  map_pars );
        }
        map_pars = dev.malloc_device< int_T > ( capacity );
        dev.sync_stream( mod.stream );

        this->   par =    par;
        this-> n_par =  n_par;
        return  true ;
    };

    __host__ virtual void reserve_host
    ( particle::base_t & mod, const int_T & n_par )
    {
        this->n_par = n_par;
        par   = mod.p_dev->malloc_host< char >
              ( n_par * get_self(  ).unit_size(  ) );
        return;
    };
    __host__ virtual void free_host
    ( particle::base_t & mod )
    {
        return mod.p_dev->free_host( par );
    }

    ////////// Data access //////////
    __host__ size_t size(  ) const
    {
        return n_par * get_self(  ).unit_size(  );
    };

    __host__ size_t n_par_eff(  ) const
    {
        return n_par - offset;
    };

    __host__ char * pdata( const int_T & offset_extra = 0 )
    {
        return ( par + ( offset + offset_extra )
                 * get_self(  ) .unit_size(  ) );
    };

    __host__ virtual void set_recv
    ( particle::base_t & mod, const int_T & n_recv )
    {
        this->n_recv = n_recv;
        offset       = n_par ;
        set_mem( mod , n_par + n_recv );
        return;
    };

    template< class pols_T > __host__ void merge
    ( const pols_T & pools, particle::base_t & mod )
    {
        size_t npar_new ( 0 );
        for( const auto & pool : pools )
             npar_new  += pool . n_par ;
        set_mem( mod, n_par + npar_new );

        size_t     ipar_new ( get_self(  ).size(  ) );
        for( const auto & p : pools )
        {
            mod.p_dev->cp_a( par + ipar_new, p.par,
                             p.size(  ), mod.stream );
            ipar_new += p.size(  );
        }
        return;
    };

    __device__ __host__ __forceinline__
    par_T & operator [  ] ( const int_T & i ) const
    {
        return ( * reinterpret_cast< par_T * >
               ( par + i * get_self(  ).unit_size(  ) ) );
    };

    __device__ __forceinline__
    int_T load( par_T & par, bool & flag ) const
    {
        const auto i = utils::th_id< int_T > (  ) + offset;
        flag   = ( i < n_par );
        if( flag )
            par.load( get_self(  )[ i ] );
        return i;
    };
 
    __device__ __forceinline__
    void mark_relocate( const int_T & i ) const
    {
        if( comm_mode == keep_pool )
            map_pars[ utils::atomic_inc( i_mv ) ] = i;
        return;
    };

    ////////// Regularize the particles //////////
    __host__ virtual void pre_proc( particle::base_t & mod )
    {
        offset = 0;
        if( mod.comm_mode(  ) == refresh_pool )
            n_par = 0 ;
        if( i_mv != nullptr )
            mod.p_dev->f_mset
              ( i_mv, 0, 2 * sizeof( int_T ), mod.stream );
        return;
    };

    template< class mod_T >
    __host__ void regulate( mod_T & mod )
    {
        auto & dev( * mod.p_dev );
        dev.sync_stream( mod.stream );
        if( mod.comm_mode(  ) != keep_pool || n_par < 1 )
            return;

        dev.cp_a( n_mv_h, i_mv, 1, mod.stream );
        dev.sync_stream          ( mod.stream );
        const auto n_mv( * n_mv_h );
        if( n_mv < 1 ) // n_mv for the most pessimistic case
            return;

        const int_T n_keep( n_par - n_mv );
        const int_T n_th( utils::min( this->n_th , n_mv ) );
        const int_T n_bl( ( n_mv + n_th  -  1  ) / n_th );
        const auto  lpar( std::make_tuple
                        ( dim3( n_bl ), dim3( n_th ), 0 ) );
        dev.launch( ker_pool_relocate, lpar, mod.stream,
                    ( * this ), n_keep,  n_mv );
        n_par = n_keep;
        return;
    };

    __device__ __forceinline__ void relocate
    ( const int_T & n_keep, const int_T & n_mv ) const
    {
        const auto i = utils::th_id< int_T >(  );
        if( i < n_mv )
        {
            const auto i_tgt = map_pars[ i ];
            auto & src = get_self (   )[ n_par - 1 - i ];
            if( i_tgt < n_keep && src.dest.todo == to_keep )
                src.save ( get_self(  )[ i_tgt ] );
        }
        return;
    };
};                              // struct    pool_t
};                              // namespace particle
