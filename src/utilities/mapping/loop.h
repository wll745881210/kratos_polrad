#pragma once

#include "../../types.h"
#include <type_traits>

namespace utils
{
////////////////////////////////////////////////////////////
// Types

using type::idx_t;

////////////////////////////////////////////////////////////
// Looping

class loop_t
{
    ////////// Initializer //////////
public:
    loop_t(  ) : include_max( false )
    {
        iter.p = this;
    };
    template< typename U, typename V >
    loop_t( const U & i_min, const V & i_max ,
            const bool include_max   = false ) : loop_t(  )
    {
        set_lim( i_min, i_max, include_max );
    };
    template<  typename U, typename V >
    inline void set_lim
    ( const U  & i_min, const V & i_max ,
      const bool include_max    = false )
    {
        this->include_max = include_max;
        for( int n = 0; n < 3; ++ n )
        {
            this->i_min[ n ] = i_min[ n ];
            this->i_max[ n ] = i_max[ n ]
                             + ( include_max ? 1 : 0 );
        }
        this->i     = this->i_min;
        is_finished =       false;
        return;
    };
    inline void reset(  )
    {
        for( int n  = 0; n < 3; ++ n )
            i[ n ]  = i_min[ n ];
        is_finished = false;
        return;
    };

    ////////// Status //////////
protected:                      // Data
    bool  include_max;
    bool  is_finished;
    idx_t       i_min;
    idx_t       i_max;
    idx_t           i;
public:                         // Functions
    inline idx_t & idx(  )
    {
        return i;
    };
    inline void operator ++ (  )
    {
        ++ i[ 0 ];
        for( int n = 0; n < 2; ++ n )
            if( i[ n ] >=  i_max[ n ] )
            {
                i[ n ]  =  i_min[ n ];
                ++ i[ n +  1 ];
            }
        if( i[ 2 ] >= i_max[ 2 ] )
            is_finished = true;
        return;
    };
    inline const bool & finished(  )
    {
        return is_finished;
    };

    ////////// Mapping //////////
public:
    template< typename F >
    inline void map          ( const F & f )
    {
        for( ; ! this->is_finished; ++ ( * this ) )
            f( i );
        return;
    };
    template< typename F >
    inline void operator(  ) ( const F & f )
    {
        this->map( f );
        return;
    };

    ////////// Range-based for loop //////////
protected:
    struct iter_t
    {
        loop_t * p;
        inline bool operator != ( const iter_t & r ) const
        {
            return ( ! p->finished (  ) );
        };
        inline void    operator ++ (  )
        {
            ++ ( * p );
        };
        inline idx_t & operator *  (  )
        {
            return p->idx(  );
        };
    };
    iter_t iter;
public:
    inline const iter_t & begin(  )
    {
        reset(    );
        return iter;
    };
    inline const iter_t &   end(  ) const
    {
        return iter;
    };
};

////////////////////////////////////////////////////////////
// Generators for loop_t

template< class V, std::enable_if_t
        < ! std::is_integral< V >::value, int > = 0 >
inline loop_t loop( const V  & i_min, const V & i_max ,
                    const bool include_max    = false )
{
    return loop_t( i_min, i_max, include_max );
}

template< typename V >
inline loop_t loop( const V  &       i_max,
                    const bool include_max    = false )
{
    return loop_t( idx_t::null(  ), i_max, include_max );
}

inline loop_t loop( const idx_t &    i_max,
                    const bool include_max    = false )
{
    return loop_t( idx_t::null(  ), i_max, include_max );
}

template< class I, std::enable_if_t
        <   std::is_integral< I >::value, int > = 0 >
inline loop_t loop( const I & i_min, const I  & i_max ,
                    const bool include_max    = false )
{
    idx_t i_mi_ ( { i_min, i_min, i_min } );
    idx_t i_ma_ ( { i_max, i_max, i_max } );
    return loop_t ( i_mi_, i_ma_, include_max );
}

};                              // namespace utils
