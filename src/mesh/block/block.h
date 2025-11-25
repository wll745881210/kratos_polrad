#pragma once

#include "../../io/binary/binary_io.h"
#include "../geometry/geometry.h"
#include <string>

namespace mesh
{
////////////////////////////////////////////////////////////
// Basic block data holder without auxilliary data

class   mesh_t;

struct block_t
{
    ////////// Identifier //////////
    int        id_g;              // Global    ID
    int        id_l;              // Local     ID
    int        rank;              // Process rank
    geo_t       geo;              // Geometries
    region_t    reg;              // Logic region

    ////////// Data //////////
    mesh_t  *                                    p_mesh;
    lex_map_t< neighbor_t, region_logic_t >   neighbors;

    ////////// Functions //////////
    block_t(  );
    virtual bool updated_neighbor(              ) const;
    virtual void read ( binary_io::base_t & bio ,
                        device   ::base_t & dev ) ;
    virtual void write( binary_io::base_t & bio ) const;
    virtual std::string        io_prefix(       ) const;
};
};                              // namespace mesh
