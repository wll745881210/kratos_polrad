#pragma once

#include "../types/driver_base.h"

namespace particle::generate
{
////////////////////////////////////////////////////////////
// Kernel function template

template< class gen_T ,    class pol_T, class map_T >
__global__ void ker_gen
( const gen_T f_gen, const pol_T  pool, const map_T bmap )
{
    return f_gen( pool, bmap );
}

////////////////////////////////////////////////////////////
// Particle generator

template < class derived_T = crtp::dummy_t >
struct base_t : ::particle::driver::base_t
{
    ////////// Type //////////
    __crtp_def_self__( base_t, derived_T );
    using super_t = ::particle::driver::base_t;
    
    ////////// Data //////////
    int           step_max;
    size_t           n_par;        
    type::float_t    f_par;
    
    ////////// Host-side interfaces //////////
    __host__ virtual void init( const      input & args ,
                                particle::base_t &  mod )
    {
        step_max = args.get< type::  int_t > 
                 ( "particle", "step_max", 100 );
        n_par    = args.get< type::float_t >
                 ( "particle",    "n_par", 1.0 );
        f_par    = args.get< type::float_t >
                 ( "particle",    "f_par", 1.0 );
        return super_t::init ( args,       mod );
    };
    template< class pol_T >
    __host__ std::tuple< dim3, dim3, int >
    resource( const pol_T & pool ) const
    {
        dim3 n_bl = ( pool.n_par + n_th - 1 ) / n_th;
        return std::make_tuple( n_bl, dim3( n_th ), 0 );
    };
    template< class pol_T, class  map_T > __host__
    void generate ( pol_T & pool, const map_T & bmap,
                    particle::base_t  & mod )
    {
        const auto & f_gen = get_self(   );
        pool.set_mem( mod, n_par );
        if( n_par > 0 )
            mod.p_dev->launch
                ( ker_gen, f_gen.resource( pool ),
                  mod.stream, f_gen, pool, bmap );
        return;
    };
};
};                         // namespace particle::generate
