#pragma once

#include "../../../io/args/input.h"
#include "../../../mesh/mesh.h"
#include "../../../utilities/functions.h"
#include "../../../utilities/types/crtp.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Enum and forward declaration
enum  comm_mode_t : int{ refresh_pool, keep_pool };
class base_t;
};

namespace particle::driver
{
////////////////////////////////////////////////////////////
// Generic particle driver interface

struct base_t
{
    ////////// Data //////////
    int        n_th ;

    ////////// Functions //////////
    __host__ virtual void init
    ( const  input & args, particle::base_t & mod )
    {
        n_th = args.get< int >( "particle", "n_th", 64 );
    };
    __host__ virtual void finalize( particle::base_t & ){ };
    __host__ virtual void evolve  ( particle::base_t & ){ };
    __host__ virtual void update
    ( const mesh::mod_base_t::v_reg_t & ,
      particle::               base_t & ) {  };
};

};                         // namespace particle::driver
