#include "svd.h"
#include <algorithm>
#include <numeric>

namespace utils::math
{
////////////////////////////////////////////////////////////
// SVD interface w/ Jacobi rotation

template < class v_T >
std::vector < size_t > argsort_descend( const v_T & v )
{
    std::vector< size_t >     idx( v.size(    ) ) ;
    std::iota( idx.begin(  ), idx.end(  ),    0 ) ;
    std::sort( idx.begin(  ), idx.end(  ),    [ & ]
             ( const size_t & i, const size_t & j )
             { return      v[ i ]    > v[ j ] ; } );
    return idx;
}

std::tuple< matrix_t <  >, matrix_t<  >, vector_t<  > >
svd( const  matrix_t <  >  & a )
{
    static const type::float2_t atol( 1e-17 );
    static const type::float2_t rtol( 1e-17 );    
    static const int            imax( 1000  );

    if( a.m == 0 || a.m > a.n )
        throw std::runtime_error( "svd.h" );

    vector_t<  >           s( a.n );
    matrix_t<  > u( a.n ), v( a.n );
    for( size_t i = 0; i < a.n ; ++ i )
        for( size_t j = 0; j < a.n; ++ j )
        {
            v( i, j ) = ( i == j ) ;
            u( i, j ) = ( i <  a.m ? a( i, j ) : 0 );
        }

    for( int iter = 0; iter < imax; ++ iter )
    {
        bool converged( true );
        for( size_t j = 1; j < a.n; ++ j )
            for( size_t i = 0; i < j; ++ i )
            {
                const auto &  ui = u .col(  i );
                const auto &  uj = u .col(  j );
                const auto &  vi = v .col(  i );
                const auto &  vj = v .col(  j );
                const auto alpha = ui.dot( ui );
                const auto beta  = uj.dot( uj );
                const auto gamma = ui.dot( uj );

                type::float2_t s( 0 ), c( 1 ), rho, rho_2;
                if( fabs( gamma ) > alpha * rtol )
                {
                    rho   = ( beta - alpha ) / ( 2 * gamma );
                    rho_2 = pow( rho, 2 );
                    const auto t =  ( rho > 0 ? 1. : -1.  ) /
                        ( fabs( rho ) + sqrt( 1 + rho_2 ) ) ;
                    c = 1 / sqrt( 1. + pow( t, 2 ) );
                    s = c * t; // sin/cos of Jac rot
                }
                if( fabs( rho ) > 1 / atol )
                    continue;                
                if( std::isnan( c + s ) )
                    throw std::runtime_error( "svd err" );
                if( converged && fabs( gamma ) > atol )
                    converged &= fabs( gamma )
                              <  sqrt( alpha * beta ) * rtol;

                for( size_t k = 0; k < a.m; ++ k )
                {
                    u( k, i ) = c * ui[ k ] - s * uj[ k ];
                    u( k, j ) = s * ui[ k ] + c * uj[ k ];
                }
                for( size_t k = 0; k < a.n; ++ k )
                {
                    v( k, i ) = c * vi[ k ] - s * vj[ k ];
                    v( k, j ) = s * vi[ k ] + c * vj[ k ];
                }
            }
        if( converged )
            break;
    }
    for( size_t j = 0; j < a.n; ++ j )
        s[ j ] = sqrt( u.col( j ).norm2(  ) );
    for( size_t i = 0; i < a.m; ++ i )
        for( size_t j = 0; j < a.n; ++ j  )
            if( s[ j ] > 0 )
                u( i , j ) /= s [ j ];
    
    const auto     & idx = argsort_descend( s );
    u.col_rearrange( idx );
    v.col_rearrange( idx );
    s.    rearrange( idx );
    s.    resize   ( a.m );
    u.    resize   ( a.m );
    return std::make_tuple( u, v, s );
}

};                              // utils::svd
