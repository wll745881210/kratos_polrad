#include "l_system.h"
#include <iostream>

namespace utils
{
namespace l_system
{
////////////////////////////////////////////////////////////
// Constructor and initalizer

l_sys_t::l_sys_t(  ) : l_lim( 0 ), regularize_ref( false )
{
    using q_t   = quat_t< int >;
    rule_ref    = [  ] ( node_t & n ) 
    {           return   false;     };    
    cons[ 'F' ] = [  ] ( node_t & n )
    {
        n.q.try_normalize(  );
        auto d = n.q * q_t( { 0, 1, 0, 0 } ) * n.q.conj(  );
        const auto q_norm = n.q.norm2(  );
        if( q_norm > 0 )
            d /= q_norm;
        for( int a = 0; a < n.x.size(  ); ++ a )
            n.x[ a ] += d[ a + 1 ];
    };
    cons[ '+' ] = [  ] ( node_t & n )
    {
        n.q *= q_t( { 1, 0, 0,  1 } );
    };
    cons[ '-' ] = [  ] ( node_t & n )
    {
        n.q *= q_t( { 1, 0, 0, -1 } );
    };
    cons[ '&' ] = [  ] ( node_t & n )
    {
        n.q *= q_t( { 1, 0,  1, 0 } );
    };
    cons[ '^' ] = [  ] ( node_t & n )
    {
        n.q *= q_t( { 1, 0, -1, 0 } );
    };
    cons[ '>' ] = [  ] ( node_t & n )
    {
        n.q *= q_t( { 1,  1, 0, 0 } );
    };
    cons[ '<' ] = [  ] ( node_t & n )
    {
        n.q *= q_t( { 1, -1, 0, 0 } );
    };
    return;
};

////////////////////////////////////////////////////////////
// 

void l_sys_t::step ( node_t & n )
{
    if( n.l >= l_lim )
        refine_attempt( n ) ;
    else
        for( const auto & r : rule[ n.a ] )
            if     ( rule.find( r ) != rule.end(  ) )
            {
                n.a    = r;
                n.l   += 1;
                step ( n );
                n.l   -= 1;
            }
            else if( cons.find( r ) != cons.end(  ) )
                cons[ r ]( n );
            else
                throw std::runtime_error( "Invalid rule" );
    return;
}

////////////////////////////////////////////////////////////
// Refinement

void l_sys_t::refine_attempt( node_t & n )
{
    if( regularize_ref || ! rule_ref ( n ) )
    {
        nodes.push_back( n );
        return;
    }
    auto i = nodes.end(  );
    --   i ;
    regularize_ref  = true;
    ++               l_lim;

    auto dx = type::idx_t::null(  );
    auto  m = n;
    m.x     = dx;
    step( m );
    for( ++ i; i != nodes.end(  ); i = nodes.erase( i ) )
        for( int a = 0; a < m.x.size(  );  ++  a )        
            dx[ a ] = std::min( dx[ a ], i->x[ a ] );
    m    = n ;      //  Copy q and a from n
    for( int a = 0; a < m.x.size(  ); ++  a )
        m.x[ a ] =  2 * n.x[ a ]    - dx[ a ];

    regularize_ref = false;    
    step( m );
    --               l_lim;
    return;
}

////////////////////////////////////////////////////////////
// Interfaces

void l_sys_t::generate( const int & lvl )
{
    l_lim = lvl;
    node_t  n_0;
    using   q_t = decltype(  n_0. q );
    n_0 .   a = rule.begin(  )->first;
    n_0 .   q = q_t( { 1, 0, 0, 0 } );
    step(   n_0 );
    return;
}

};                              // namespace l_system
};                              // namespace utils
