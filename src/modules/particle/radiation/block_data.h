#pragma once

#include "../../../mesh/block/block_data.h"
#include "rad_data.h"

namespace particle::radiation
{
////////////////////////////////////////////////////////////
// Block data holder for hydrodynamics

using namespace mesh;

struct block_data_t : mesh::block::base_data_t
{
    ////////// Type and Data //////////
    rad_t      rad;

    ////////// Memory operations //////////
    __host__ virtual void setup( const f_new_t  & f_n )
    {
        return rad.setup( f_n, geo );
    };
    __host__ virtual void free ( const f_free_t & f_f )
    {
        return rad.free( f_f );
    };

    ////////// Data transfer //////////
    __host__ virtual void copy_input
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return rad.copy_to
        ( f_cp, static_cast< block_data_t & >( tgt ). rad );
    };
    __host__ virtual void copy_output
    ( const f_cp_t & f_cp, mesh::block::base_data_t & tgt )
    {
        return copy_input( f_cp, tgt );
    };
    __host__ virtual void read ( const f_read_t  & f_r )
    {
        return rad.read( f_r );
    };
    __host__ virtual void write( const f_write_t & f_w )
    {
        return rad.write( f_w );
    };
};

};                        // namespace particle::radiation
