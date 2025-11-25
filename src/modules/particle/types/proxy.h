#pragma once

#include "driver_base.h"
#include "particle_base.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Block proxies

struct proxy_base_t
{
    struct
    {
        int               id;
        int           i_rank;
        bool      shift[ 2 ];
    }   neighbors [ 3 ][ 2 ][ 2 ][ 2 ];
    type::schar_t d_lvl[ 6 ];
    mesh::  geo_t   geo;

    __host__ virtual void setup
    ( const mesh::block::dual_t   &      d ,
      const std ::map< int, int > & m_rank ,
      particle  ::base_t          &    mod )
    {
        d.d(  ).geo.cp_shlw( geo );
        for( auto & nb_s : neighbors )
            for( auto & nb_ss : nb_s )
                for( auto & nb_sss : nb_ss  )
                    for( auto & nb : nb_sss )
                    {
                        nb.    id = -1;
                        nb.i_rank = -1;
                    }
        for( auto & dl : d_lvl )
            dl = 0;

        const auto &  b ( d.b_w.get(  ) );
        for( const auto & [ nb, reg ] : b.neighbors )
        {
            if( nb.mode != 0 )
                continue;
            for( int a = 0; a < 3; ++ a )
            {
                if( nb[ a ] >= 0 && nb[ a ] <= 1 )
                    continue;
                auto & n = neighbors[ a ] [ nb[ a ] > 0 ]
                                    [ nb[ ( a + 1 ) % 3 ] ]
                                    [ nb[ ( a + 2 ) % 3 ] ];
                const auto &  g = nb.guest_info;
                if( g.rank  != b.rank ) // Skip same-rank
                    n.i_rank = m_rank.at( g.rank );
                n.id = g.id_l;
                d_lvl[ 2 * a + ( nb[ a ] > 0 ) ] = nb.d_lvl;
                const auto & nb_r ( nb.reverse (  ) );
                for( int i = 0; i <= 1;  ++  i )
                   n.shift[ i ] = nb_r[ ( a + i + 1 ) % 3 ];
            }
        }
        return;
    };

    template < class par_T > __forceinline__
    __device__ bool shift_blk( par_T & par ) const
    {
        par.dest.todo = particle::to_keep;
        auto & i( par.i );
        const auto & n_ceff( geo.n_ceff );
        for( int a = 0; a < 3; ++ a )
        {
            if( i[ a ] >= 0 && i[ a ] < n_ceff[ a ] )
                continue;
            if( ( i[ a ] < 0 && geo.phybnd_flag[ a ][ 0 ] )
             || ( i[ a ] >= n_ceff[ a ] &&
                   geo.phybnd_flag[ a ][ 1 ] ) )
            {
                par.dest.todo = particle::to_rm;
                break;
            }
            
            const int  ap[ 2 ] = { utils::mod3( a + 1 ) ,
                                   utils::mod3( a + 2 ) } ;
            const auto dl = d_lvl[ 2 * a + ( i[ a ] > 0 ) ];
            int i_na[ 2 ] = { 0, 0 };
            if( dl  > 0 )
                for( int  j = 0;  j <= 1 ; ++  j )
                    i_na[ j ] = ( i      [ ap[ j ] ] >=
                                  n_ceff [ ap[ j ] ] / 2 );
            const auto & nb = neighbors[ a ][ i[ a ] > 0 ]
                              [  i_na[ 0 ] ][ i_na [ 1 ] ];

            par.ib_l = nb.id;  // All-purpose !
            if( nb .i_rank >= 0 )
                par.dest.i_rank =        nb.i_rank;
            else if ( nb.id < 0 )
            {
                par.dest  .todo =  particle::to_rm;
                break;
            }
            i[ a ] = ( i[ a ] >  0 ? 0 : n_ceff[ a ] - 1 );

            if( dl < 0 )
                for( int j = 0 ; j <= 1; ++ j )
                {
                    auto & is( i[ ap[ j ] ] );
                    is = is / 2 + nb.shift  [ j ]
                                * n_ceff[ ap[ j ] ] / 2;
                }
            else if( dl > 0 )
                for( int j = 0; j <= 1;  ++ j )
                {
                    auto & is( i[ ap[ j ] ] );
                    is = ( is < n_ceff[ ap[ j ] ] / 2 ? is :
                           is - n_ceff[ ap[ j ] ] / 2 ) * 2;
                }
            break;
        }
        return ! ( par.dest.  todo == particle::to_rm ||
                   par.dest.i_rank >= 0 );
    };
};
};                              // namespace particle
