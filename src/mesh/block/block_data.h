#pragma once

#include "../../types.h"
#include "../geometry/geometry.h"
#include <functional>

namespace mesh::block
{
////////////////////////////////////////////////////////////
// Blocks (sub-domains) in the mesh

struct base_data_t
{
    ////////// Data //////////
    int       output_flag;
    type ::float_t      t;
    type ::float_t     dt;
    type ::float_t   dt_0;  // Full dt without substepping
    type ::float_t * p_dt;  // For device-side dt calc
    region_logic_t    reg;
    geo_t             geo;

    ////////// Derived access //////////
    template< class dat_T >
    __host__  dat_T & derive(  ) const
    {
        return dynamic_cast< dat_T       & >
             ( * const_cast< base_data_t * > ( this ) );
    };      // dynamic_cast is the last line of defense!

    ////////// Functions //////////
    __host__ virtual void setup( const   f_new_t & ) {  };
    __host__ virtual void free ( const  f_free_t & ) {  };
    __host__ virtual void copy_input
    ( const  f_cp_t  &,   base_data_t   &  )         {  };
    __host__ virtual void copy_output
    ( const  f_cp_t  &,   base_data_t   &  )         {  };
    __host__ virtual void read ( const  f_read_t & ) {  };
    __host__ virtual void write( const f_write_t & ) {  };
};
};                              // namespace mesh::block
