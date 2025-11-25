#include "../../utilities/functions.h"
#include "boundary.h"

namespace mesh::boundary
{
////////////////////////////////////////////////////////////
// Base boundary opeartions for comm boundaries

__host__   std::tuple< int , int >
comm_base_t::resource_thbl( const idx_t & n_cell ) const
{
    const int n_th
        = std::min( n_cell[ axis[ 1 ] ], this ->n_th_lim );
    int n_bl = ( n_th - 1 + n_cell[ axis[ 1 ] ] ) / n_th  ;
    return std::make_tuple( n_th, n_bl );
};

__host__ void comm_base_t::setup
( holder_c_t & h, const neighbor_t & nb, mod_base_t & mod )
{
    set_geo   ( h, nb, mod );
    set_mem   ( h, nb, mod );
    set_comm  ( h, nb, mod );
    set_local ( h, nb, mod );
    set_launch( h, nb, mod );
    return;
}

__host__ void comm_base_t::set_geo
( holder_c_t & h, const neighbor_t & nb, mod_base_t & mod )
{
    this->mode  = nb.mode;
    this->d_lvl = h.d_lvl;
    auto  & geo = h.p_d->b_w.get(  ).geo;
    this->rhs   = 0;
    this->same_rank  = h.same_rank(  );
    for( int  a = 0; a < axis.size(  ); ++  a )
    {
        axis[ a ] = 0;
        if( ( mode != 0 ) ^ ( nb[ a ] < 0 || nb[ a ] > 1 ) )
        {
            rhs = ( nb[ a ] > 0 ) ;
            for( int n = 0; n < 3 ; ++ n )
                axis[ ( mode + n ) % geo.n_dim ]
                    = ( a    + n ) % geo.n_dim ;
            break;
        }
        else if( mode == 2 )
            axis[ a ]  = a ;
        //  axis[ 0 ] norm for mode 0;
        //  axis[ 1 ] extension for both modes 0 and 1
    }   //  axis[ a ] almost arbitrary for mode 2
    this->offset_s = idx_t::null(  );
    this->offset_r = idx_t::null(  );
    this->offset_c = idx_t::null(  );
    return;
}

////////////////////////////////////////////////////////////
// Boundary holder

holder_c_t::holder_c_t( block::dual_t & d )
{
    p_d = ( & d );
    return;
}

bool holder_c_t::same_rank(  ) const
{
    return p_d->b_w.get(  ).rank == rank;
}

void holder_c_t::setup
( const neighbor_t & nb, mod_base_t & mod )
{
    auto  & b ( p_d->b_w.get(  ) );
    d_lvl = nb.              d_lvl;
    rank  = nb. guest_info.   rank;
    if( ! same_rank(  ) )
    {
        const int w = 2 * nb.size(  ) + 1;
        tag_s = ( b.id_l << w ) + ( nb.tag(  ) << 1 );
        tag_r = ( nb.guest_info.id_l << w )
              + ( nb.reverse      (  ).tag(  ) << 1 );
        tag_f = ( d_lvl < 0 ? tag_s :  tag_r ) +  1 ;
    }   // Separated flux comm tag to avoid tag collision
    return p->setup( * this, nb, mod );
}

};                              // namespace mesh::boundary
