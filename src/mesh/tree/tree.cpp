#include "../../utilities/mapping/loop.h"
#include "tree.h"

#include <iostream>

namespace mesh
{
////////////////////////////////////////////////////////////
// IO

std::ostream & tree_t::write( std::ostream & stream )  const
{
    using base_t = lex_map_t< region_logic_t, tree_node_t >;
    auto & base = static_cast< const base_t & >  ( * this );
    serialize::write ( stream,       base ) ;
    serialize::write ( stream,      n_dim ) ;
    serialize::write ( stream,    reg_max ) ;
    serialize::write ( stream,   periodic ) ;
    return stream;
};

std::istream & tree_t::read ( std::istream & stream )
{
    using base_t = lex_map_t< region_logic_t, tree_node_t >;
    auto & base = static_cast<       base_t & >  ( * this );
    serialize::read  ( stream,       base ) ;
    serialize::read  ( stream,      n_dim ) ;
    serialize::read  ( stream,    reg_max ) ;
    serialize::read  ( stream,   periodic ) ;
    return stream;
};

////////////////////////////////////////////////////////////
// Size

size_t tree_t::size_eff(  ) const
{
    size_t count( 0 );
    for  ( const auto & p  : ( * this ) )
           count += p.second.mask ? 0 : 1;
    return count;
}

size_t tree_t::max_lvl (  ) const
{
    return size_t( this->rbegin(  )->first.level );
}

////////////////////////////////////////////////////////////
// Inquires about the global properties

bool tree_t::contains( const region_logic_t & reg ) const
{
    return this->find( reg ) != this->end(  );
}

idx_t tree_t::region_max( const int & level ) const
{
    auto  res = reg_max;
    for( auto & r_s : res.x )
        r_s <<= level;
    return res;
}

////////////////////////////////////////////////////////////
// Utilities

bool tree_t:: is_inside( const region_logic_t & reg ) const
{
    const auto rl = region_max( reg.level );
    bool  res( true );
    for(  int n = 0; n < n_dim; ++ n )
          res &=  ( reg[ n ] >= 0 && reg[ n ] < rl[ n ] );
    return res;
}

void tree_t::regularize(       region_logic_t & reg ) const
{
    const auto rl = region_max( reg.level );
    for( int n = 0; n < n_dim; ++ n )
        if( periodic[ n ] )
            reg[ n ] = ( reg[ n ] + rl[ n ] ) % rl[ n ];
    return;
}

////////////////////////////////////////////////////////////
// Interface mapping functions to region-related things

void tree_t::map_refined
( const f_reg_t & f_reg, const region_logic_t & reg ) const
{
    auto di = idx_t::null(  );
    f_int_t f_dim_recursive = [ & ] ( const int & a )
    {
        if( a < n_dim )
        {
            for( di[ a ] = 0; di[ a ] < 2; ++ di[ a ] )
                f_dim_recursive ( a + 1 );
            --   di[ a ];
            return;
        }
        auto  r  = reg;
        r.level  = reg.level + 1;
        for( int n = 0;  n < n_dim;  ++ n )
            r[ n ] = di[ n ] + 2 * reg[ n ];
        f_reg( r ) ;
    };
    f_dim_recursive( 0 );
    return;
}

void tree_t::map_refined_nb
( const f_reg2_t & f_reg, const region_logic_t & reg ) const
{
    auto di = idx_t::null(  ) ;
    f_int_t f_dim_recursive = [ & ] ( const int & a )
    {
        if( a < n_dim )
        {
            for( di[ a ] = -1; di[ a ] < 3; ++ di[ a ] )
                f_dim_recursive  ( a + 1 );
            --   di[ a ];
            return;
        }
        bool inside_blk( true );
        for( int n = 0; n < n_dim; ++ n )
            inside_blk &= ( di[ n ] >= 0 && di[ n ] <= 1 );
        if( inside_blk )
            return;

        auto tst =  reg;
        tst. level += 1;
        for( int n = 0;    n < n_dim;  ++ n )
            tst[ n ] = di[ n ] + 2 * reg[ n ];
        const auto orig = tst   ;
        regularize      ( tst ) ;
        if( is_inside   ( tst ) )
            f_reg( orig , tst ) ;
    };
    f_dim_recursive( 0 );
    return;
}

void tree_t::map_neighbor
( const f_reg_t & f_reg, const region_logic_t & reg ) const
{
    auto di = idx_t::null(  ) ;
    f_int_t f_dim_recursive = [ & ] ( const int & a )
    {
        if( a < n_dim )
        {
            for( di[ a ] = -1; di[ a ] < 2; ++ di[ a ] )
                f_dim_recursive  ( a + 1 );
            --   di[ a ];
            return;
        }
        bool inside_blk( true );
        for( int n = 0; n < n_dim ; ++ n )
            inside_blk &= ( di[ n ] == 0 );
        if( inside_blk )
            return;

        auto tst = reg;
        for( int n = 0;  n < n_dim; ++ n )
            tst[ n ] = reg [ n ] + di[ n ];
        regularize   ( tst ) ;
        if( is_inside( tst ) )
            f_reg    ( tst ) ;
    };
    f_dim_recursive( 0 );
    return;
}

////////////////////////////////////////////////////////////
// Get all existing neighbors

lex_map_t< neighbor_t, region_logic_t >
tree_t  :: neighbors ( const region_logic_t & reg ) const
{
    std::map< neighbor_t, region_logic_t, lex_ord_t > res;
    const int l_min = std::max( 0, reg.level - 1 );
    map_refined_nb( [ & ]( const auto & r0, const auto & r )
    {
        for( auto tst = r   , ts0 = r0; tst.level >= l_min;
             tst.coarsen(  ), ts0.coarsen(  ) )
            if( contains( tst ) && ! at ( tst ).mask )
            {
                neighbor_t nb     ( reg,     ts0 );
                nb.guest_reg      = nb .   guest  ;
                regularize        ( nb.guest_reg );
                nb.guest_info = at( nb.guest_reg );
                res[ nb ]     =              tst  ;
                break;
            }
    },  reg );
    return  res;
}

////////////////////////////////////////////////////////////
// Mesh management

void tree_t::insert( const region_logic_t & reg )
{
    if( contains( reg ) )
        return;
    if( reg.level  >  0 )
    {
        const auto reg_c     = reg.coarser (  );
        insert( reg_c );
        at    ( reg_c ).mask = true;
        map_refined   ( [ & ]( const auto & r )
        {   ( * this  ) [ r ];  },      reg_c );
        map_neighbor  ( [ & ]( const auto & r )
        {     insert    ( r );  },      reg_c );
    }
    ( * this )  [ reg ];
    return;
}

void tree_t::fill_base_level(  )
{
    region_logic_t    reg;
    for( const auto & i  : utils::loop( reg_max ) )
    {
        reg       =     i;
        reg.level =     0;
        ( * this )[ reg ];
    }
    return;
}

};                              // namespace mesh
