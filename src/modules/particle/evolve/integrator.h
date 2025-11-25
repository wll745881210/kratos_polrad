#pragma once

#include "../types/particle_base.h"
#include "../types/driver_base.h"

namespace particle::integrate
{
////////////////////////////////////////////////////////////
// Kernel interface for the integrator

template< class itg_T, class pol_T ,
          class map_T, class com_T > __global__
void ker_itg( const itg_T f_itg, const pol_T pool ,
              const map_T  bmap, const com_T comm )
{
    return f_itg( pool, bmap, comm );
}

////////////////////////////////////////////////////////////
// Integrate the particles

template  < class derived_T = crtp::dummy_t >
struct base_t : ::particle::driver:: base_t
{
    ////////// Type //////////
    __crtp_def_self__( base_t, derived_T );
    int     round;
    int max_round;

    ////////// Host-side interfaces //////////
    __host__ virtual void init
    ( const  input & args, particle::base_t & mod ) override
    {
        round = 0 ;
        max_round = args.get< int >
                  ( "particle", "max_round", 16 );
        return driver::base_t::init( args,  mod );
    };
        
    template < class pol_T >
    __host__ std::tuple< dim3, dim3, int >
    resource( const pol_T & pool ) const
    {
        dim3 n_bl(( pool.n_par_eff( ) + n_th - 1 ) / n_th );
        return std::make_tuple ( n_bl, dim3( n_th )  ,  0 );
    };
    template< class  pol_T, class  map_T , class com_T >
    __host__  bool    intg
    ( const   pol_T & pool, const      map_T & bmap ,
      const   com_T & comm, particle::base_t &  mod )
    {
        ++ round;
        if( pool.n_par_eff(  ) == 0 || round >= max_round )
            return false;
        for( auto & d :  mod )
            mod.p_dev->event_wait( mod.stream, d.event );
        const auto  & f_itg   ( this->get_self(  ) );
        mod.p_dev->launch( ker_itg,  f_itg.   resource
                         ( pool ),   mod.stream, f_itg ,
                           pool,           bmap,  comm );
        mod.p_dev->event_record( mod.event, mod.stream );
        return true;
    };

    template< class pol_T, class map_T, class com_T >
    __host__   void pre_proc
    ( pol_T & pool, map_T &, com_T &, particle::base_t & m )
    {
        this->round = 0;
        return pool.pre_proc( m );
    };
    template< class pol_T, class map_T, class com_T >
    __host__   void  post_proc
    ( pol_T &, map_T &, com_T &, particle::base_t & )
    {
        this->round = 0;
    };

    ////////// Device-side interfaces //////////
    template< class pol_T, class map_T, class com_T >
    __device__ __forceinline__    void  operator (  )
    ( const pol_T & pool , const map_T  & bmap,
      const com_T & comm ) const
    {
        bool flag;
        typename pol_T::par_t par;
        __dyn_shared__( char, p_sh );
        par  .init          ( p_sh );
        
        const auto i_par = pool.load( par, flag );
        if( ! flag ||  par.dest.todo != to_keep )        
            return ;
        const auto i_rank  = par.proc( bmap, get_self(  ) );
        if( par.dest.todo != to_rm )
        {
            par.save( pool[ i_par ] );            
            if( i_rank >=  0 )
            {
                comm.reg   ( i_par, i_rank );
                pool.mark_relocate( i_par  );
            }
        }
        else
            pool.mark_relocate    ( i_par  );
        return;
    };
};                       // struct base_t
};                       // namespace particle::integrate
