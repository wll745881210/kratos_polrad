#pragma once

#include "../../types.h"
#include <vector>

namespace utils::math
{
////////////////////////////////////////////////////////////
// Vector and matrix types

template< class f_T = type::float2_t >
struct vector_t : public std::vector < f_T >
{
    using std::vector< f_T >::vector; // Using constructors

    f_T dot( const vector_t< f_T > & b ) const
    {
        f_T res( 0 );
        for( size_t i = 0; i < this->size(  ); ++ i )
            res += this->at( i ) * b[ i ];
        return res;
    };
    f_T norm2(  ) const
    {
        return dot( * this );
    };
    template< class v_T > void rearrange( const v_T & idx )
    {
        std::vector< f_T > res( idx.size(  ) );
        for( size_t i = 0; i  < idx.size(  ); ++ i )
             res[ i ] = this->at( idx[ i ] );
        return this->swap( res );
    };
};

template< class f_T = type::float2_t >
struct matrix_t
{
    ////////// Type and data //////////
    using        f_t = f_T;
    size_t       m , n , s;
    std::vector< f_T > dat;

    ////////// Functions //////////
    matrix_t( const size_t & m = 0, const size_t & n = 0 )
    {
        this->m =   m ;
        this->n = ( n > 0 ? n : m );
        this->s = this->m * this->n;        
        dat.resize( this->s );
        std::fill ( dat.begin(  ), dat.end(  ), 0 );
        return;
    };
    f_T & operator(  )( const int & i, const int & j ) const
    {
        if( i < 0 || i >= m || j < 0 || j >= n )
            throw std::runtime_error( "matrix indices" );
        return const_cast< std::vector< f_T > & >( dat )
             [ size_t( i ) * n + size_t( j ) ];
    };
    matrix_t< f_T > tran(  ) const
    {
        matrix_t< f_T > res( n, m );
        for( size_t i = 0; i < res.m; ++ i )
            for( size_t j = 0; j < res.n; ++ j )
                res( i, j ) = ( * this )( j, i );
        return res;
    };
    template< class mat_T >
    matrix_t< f_T > mul ( const mat_T & b ) const
    {
        auto & a( * this );
        if( a.n != b.m )
            throw std::runtime_error( "matmul" );

        matrix_t< f_T > res ( a.m , b.n );
        for( size_t i = 0; i < res.m; ++ i )
            for( size_t j = 0; j < res.n; ++ j )
            {
                res( i, j ) = 0;
                for( size_t  k = 0; k  < this-> n; ++ k )
                     res( i, j ) += a( i, k ) * b( k, j );
            }
        return res;
    };
    vector_t< f_T > col( const size_t & j ) const
    {
        vector_t< f_T > res( m );
        for( size_t i = 0; i < m; ++  i )
            res[ i ] = ( * this )( i, j );
        return res;
    };
    void swap( matrix_t< f_T > & m )
    {
        auto f_swap = [ & ] ( auto & a, auto & b )
        {
            const auto c = a;
            a            = b;
            b            = c;
        };
        f_swap( this->m, m.m ) ;
        f_swap( this->n, m.n ) ;
        f_swap( this->s, m.s ) ;
        return this->dat.swap( m.dat );
    };
    template< class v_T >
    void col_rearrange( const v_T & idx )
    {
        matrix_t< f_T > res  ( this->m, idx.size(  ) );
        for( size_t i = 0; i < this->m; ++  i )
            for( size_t j = 0; j < idx.size(  ); ++  j )
                res( i, j ) =  ( * this )( i, idx[ j ] );
        return this->swap( res );
    };
    void resize( const size_t & m, const size_t & n = 0 )
    {
        matrix_t   < f_T > res  ( m, n > 0  ? n : m );
        const auto m_min = std::min( res.m, this->m );
        const auto n_min = std::min( res.n, this->n );
        for( size_t i = 0; i < m_min; ++ i )
            for( size_t j = 0; j < n_min ; ++ j )
                 res( i, j ) = ( * this )( i, j );
        return this->swap( res );
    };
};
};                              // namespace utils::math
