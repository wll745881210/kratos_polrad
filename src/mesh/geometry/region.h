#pragma once

#include "../../types.h"
#include <map>

namespace mesh
{
////////////////////////////////////////////////////////////
// Types

using type::coord_t;
using type::  idx_t;

////////////////////////////////////////////////////////////
// Lexicographical order

struct lex_ord_t
{
    template< class T , class U > __host__
    bool operator(  ) ( const T & l, const U & r ) const
    {
        if( l.level != r.level )
            return l.level < r.level;
        else
            for( int i = 0; i < l.size(  ); ++ i )
            {
                if   ( l[ i ] == r[ i ] )
                    continue;
                return l[ i ] <  r[ i ] ;
            }
        return false;
    };
};

template< class key_T, class val_T >
using lex_map_t = std::map < key_T , val_T, lex_ord_t >;

////////////////////////////////////////////////////////////
// Node types for the mesh tree

struct tree_node_t
{
    int  id_l;
    int  id_g;
    int  rank;
    bool mask;

    tree_node_t(  ) : id_l(    -1 ), id_g( -1 ), rank( -1 ),
                      mask( false )      {    };
};

////////////////////////////////////////////////////////////
// Logic and physical locations of regions

struct region_logic_t : public idx_t
{
    ////////// Data //////////
    int       level;

    ////////// The cooresponding coarser region //////////
    __host__ region_logic_t & coarsen(  )
    {
        level -= 1;
        for( auto & i_s : x )
            i_s = ( i_s > 0 ? i_s : i_s - 1 ) / 2;
        return ( * this );
    };
    __host__ region_logic_t   coarser(  ) const
    {
        region_logic_t res = ( * this );
        res.coarsen(  );
        return res;
    };

    ////////// Assignment //////////
    __host__ region_logic_t & operator = ( const idx_t & s )
    {
        idx_t::operator = ( s );
        return ( * this );
    };
};

struct region_t : public region_logic_t
{
    ////////// Data //////////
    coord_t x_mid;
    coord_t x_lim[ 2 ];

    ////////// Function //////////
    __host__ region_t &  operator =
    ( const  region_logic_t & src )
    {
        region_logic_t::operator = ( src );
        return ( * this );
    };
    template < class x_T >
    __host__ bool inside( const x_T & x )
    {
        bool res( true );
        for( int a = 0; a < 3; ++ a )
            res &= ( x[ a ] >= x_lim[ 0 ][ a ] &&
                     x[ a ] <  x_lim[ 1 ][ a ] );
        return res;
    };
};

////////////////////////////////////////////////////////////
// Neighbor; Direction flag: { -1, 0, 1, 2 }

struct neighbor_t : public region_logic_t
{
    ////////// Data //////////
    int       mode;         // Face: 0; Edge: 1; Vertex: 2
    int      d_lvl;

    tree_node_t     guest_info;
    region_logic_t        host;
    region_logic_t       guest;
    region_logic_t   guest_reg;    

    ////////// Constructor //////////
    neighbor_t( const region_logic_t & h ,
                const region_logic_t & g )
              : host( h ),   guest   ( g )
    {
        level  = 0;
        d_lvl  = g.level - h.level;
        for( int n = 0; n < size(  ); ++ n )
        {
            x[ n ] = d_lvl >= 0 ?  ( h[ n ] <<  d_lvl ) :
                                   ( h[ n ] >> -d_lvl ) ;
            x[ n ] = g[ n ]     -    x[ n ] ;
            if( x[ n ]  > 0 && d_lvl <= 0 )
                x[ n ] *= 2;
        }
        mode = -1 ;
        for( auto &    d_s :  x )
             mode += ( d_s != 0 && d_s != 1 ? 1 : 0 );
        return;
    };

    ////////// Deduce "reversed" neighbor info //////////
    neighbor_t reverse(  ) const
    {
        return neighbor_t( guest, host );
    };

    ////////// Unique integer tag //////////
    int tag(  ) const
    {
        int  res ( 0 );
        for( const  auto & i : x )
             res = ( res + i + 1 ) << 2;
        return ( res  >> 2 ) ;
    };
};

};                              // namespace mesh
