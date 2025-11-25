#pragma once

#include <vector>
#include <memory>
#include <utility>
#include <functional>

#include "../../io/args/input.h"
#include "../../utilities/l_system/l_system.h"
#include "../geometry/geometry.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Types
class     mesh_t;

namespace meshgen
{
using type::      idx_t;
using type::    float_t;
using type::    coord_t;
using type::   coord2_t;
using float2_t = double;

struct refine_region_t
{
    int      level;
    idx_t   i[ 2 ];
    coord_t x[ 2 ];
};

////////////////////////////////////////////////////////////
// Mesh generator: the default base class

class base_t
{
    ////////// Initialization //////////
public:                         // Function
    virtual void read( const input & args );
    virtual void init( const input & args , mesh_t & mesh );

    ////////// Global mesh configuration //////////
protected:                      // Data
    idx_t         n_ceff_global;
    idx_t          n_ceff_block;
    idx_t           i_logic_lim;
    int                   n_dim;
public:                        // Data
    coord2_t  x_lim_global[ 2 ];    

    ////////// Mesh tree generating //////////
protected:                      // Data
    std::vector< refine_region_t > ref_regions ;
protected:                      // Function
    virtual void load_refinement( const input & args );
    virtual void generate_tree  ( mesh_t      & mesh );

    ////////// Load balancing //////////
protected:                      // Data
    utils::l_system::l_sys_t                   l;
    std::vector  < int >                per_rank;
    std::function< void( mesh_t & ) > dist_block;
protected:                      // Functions
    virtual void dist_fractal  ( mesh_t & mesh );
    virtual void dist_default  ( mesh_t & mesh );

    ////////// Geometry generating //////////
protected:                      // Data
    bool        geo_regenerate;
protected:                      // Functions
    virtual void set_uniformity ( block_t & b );
    virtual void set_coord_axes ( block_t & b );
    virtual void set_geo_fulldim( block_t & b );
public:                         // Function
    virtual float2_t location
    ( const float2_t & x_logic, const int & axis ) const;
    virtual float2_t   surface
    ( const idx_t  &, const geo_t &, const int & ) const;
    virtual float2_t     volume
    ( const idx_t & idx,  const geo_t & geo ) const;
    virtual int  search_blk_idx
    ( const float2_t & x, const int & l, const int & ax );

    ////////// Interface //////////
protected:                      // Function
    virtual void associate_mesh ( block_t &, mesh_t & );
    virtual void create(   mesh_t & );
public:                         // Function
    virtual void save  (   mesh_t &, binary_io::base_t & );
    virtual void update(   mesh_t & );
};

};
};                               // namespace mesh::meshgen
