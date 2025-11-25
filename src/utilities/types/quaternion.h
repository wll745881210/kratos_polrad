#pragma once

#include <type_traits>
#include <algorithm>
#include <iostream>
#include <climits>

namespace utils
{
////////////////////////////////////////////////////////////
// Quaternions, including those on integers (for meshgen!)

template< class T >
struct quat_t
{
    T q[ 4 ];

    quat_t(  ) : q{ T( 0 ), T( 0 ), T( 0 ), T( 0 ) } {  };
    template< typename ... L >
    quat_t( const L & ... ls ) : q{ ls ... }         {  };

    T & operator[  ] ( const int & n ) const
    {
        return const_cast< T * >( q )[ n ];
    };
    friend quat_t< T > operator * ( const quat_t< T > & l ,
                                    const quat_t< T > & r )
    {
        quat_t< T > res;
        res[ 0 ] = l[ 0 ] * r[ 0 ] - l[ 1 ] * r[ 1 ]
                 - l[ 2 ] * r[ 2 ] - l[ 3 ] * r[ 3 ];
        res[ 1 ] = l[ 0 ] * r[ 1 ] + l[ 1 ] * r[ 0 ]
                 + l[ 2 ] * r[ 3 ] - l[ 3 ] * r[ 2 ];
        res[ 2 ] = l[ 0 ] * r[ 2 ] - l[ 1 ] * r[ 3 ]
                 + l[ 2 ] * r[ 0 ] + l[ 3 ] * r[ 1 ];
        res[ 3 ] = l[ 0 ] * r[ 3 ] + l[ 1 ] * r[ 2 ]
                 - l[ 2 ] * r[ 1 ] + l[ 3 ] * r[ 0 ];
        return  res ;
    };
    void operator *= ( const quat_t< T > & r )
    {
        ( * this ) = ( * this ) * r;
        return;
    };
    void operator /= ( const T & r )
    {
        for( int n = 0; n < 4; ++ n )
            q[ n ] /= r;
        return;
    };
    quat_t< T > conj(  )
    {
        quat_t< T > res = ( *  this );
        for( int n = 1; n < 4; ++ n )
            res.q[ n ] *= -1;
        return res;
    };
    T norm2(  )
    {
        T res( 0 );
        for( int n = 0; n < 4; ++ n )
            res += q[ n ] * q[ n ];

        return res;
    };
    void print(  )
    {
        std::cout << q[ 0 ] << ' ' << q[ 1 ] << ' '
                  << q[ 2 ] << ' ' << q[ 3 ] << '\n';
    };
    std::enable_if_t< std::is_integral< T >::value, void >
    try_normalize(  )
    {
        static const T norm_lim( INT_MAX );
        T to_norm( norm_lim );
        for( const auto & q_s : q )
            if( abs( q_s ) > 0 )
                to_norm = std::min( abs( q_s ), to_norm );
        if( to_norm < norm_lim )
            ( * this ) /= to_norm;
        return;
    };
};

};                              // namespace utils
