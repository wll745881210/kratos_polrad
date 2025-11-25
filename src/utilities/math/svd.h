#pragma once

#include "matrix.h"
#include <tuple>

namespace utils::math
{
////////////////////////////////////////////////////////////
// SVD interface w/ Jacobi rotation
// a = u * s * v', v' --> (Transpose of v)

std::tuple< matrix_t <  > , matrix_t<  >, vector_t<  > >
svd( const  matrix_t <  > & a );

template  < class  m_T >
std::tuple< matrix_t <  >, matrix_t<  >, vector_t<  > >
svd( const  m_T  & a )
{
    matrix_t <  > aa( a.m, a.n );
    for( size_t i = 0; i < a.m; ++ i )
        for( size_t j = 0; j < a.n; ++ j )
             aa( i, j ) = a( i, j );
    return svd ( aa ) ;
}
};                              // utils::svd
