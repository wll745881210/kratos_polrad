#pragma once

#include "../../types.h"
#include "../../utilities/data_transfer/serialize.h"
#include "../geometry/region.h"

#include <functional>
#include <map>
#include <set>

namespace mesh
{
////////////////////////////////////////////////////////////
// Types

using type:: idx_t;
using type::bvec_t;

using f_int_t  = std::function < void( const int & ) >;
using f_reg_t  = std::function
                    < void( const region_logic_t & ) >;
using f_reg2_t = std::function< void
  ( const region_logic_t &, const region_logic_t & ) >;

////////////////////////////////////////////////////////////
// Tree structures for blocks

struct tree_t : serialize::interface,
                lex_map_t< region_logic_t, tree_node_t >
{
    ////////// Data //////////
    int           n_dim;
    idx_t       reg_max;
    bvec_t     periodic;

    ////////// Size //////////
    size_t     size_eff(  ) const;
    size_t      max_lvl(  ) const;

    ////////// Property access //////////
    bool  contains     ( const region_logic_t & reg ) const;
    idx_t region_max   ( const int  & level = 0     ) const;

    ////////// Utilities //////////
    bool  is_inside    ( const region_logic_t & reg ) const;
    void  regularize   (       region_logic_t & reg ) const;

    ////////// Mapping interface //////////
    virtual void map_refined
    ( const f_reg_t  & f , const region_logic_t & r ) const;
    virtual void map_neighbor
    ( const f_reg_t  & f , const region_logic_t & r ) const;
    virtual void map_refined_nb
    ( const f_reg2_t & f , const region_logic_t & r ) const;

    ////////// Neighbors //////////
    virtual lex_map_t< neighbor_t, region_logic_t >
    neighbors ( const region_logic_t & reg ) const;

    ////////// Management //////////
    virtual void insert  ( const region_logic_t & reg );
    virtual void fill_base_level (                    );

    ////////// IO //////////
    std::ostream & write( std::ostream & stream ) const;
    std::istream & read ( std::istream & stream )      ;
};

};                              // namespace mesh
