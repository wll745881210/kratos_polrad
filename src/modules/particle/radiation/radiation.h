#pragma once

#include "../particle.h"
#include "block_data.h"
#include "photon.h"

namespace particle::radiation
{
////////////////////////////////////////////////////////////
// Types for block access

template< class bdt_t = block_data_t >
struct proxy_t : mesh::block::proxy_t< bdt_t >, 
                 proxy_base_t {     };

////////////////////////////////////////////////////////////
// Base type for radiation

class base_t : public particle::base_t
{
protected:                      // Data
    int     n_hv;               // # of photon energy bins
public:                         // Function
    virtual void read( const input & args )
    {
        n_hv = args.get < int > ( "particle", "n_hv", 1 );
        if( ! f_bdt_yield )
            enroll_dat < block_data_t > (    );
        return particle::base_t::read ( args );
    };
    template < class prx_T = proxy_t               <  >  ,
               class pol_T = pool_t< photon::cart_t<  > >,
               class gen_T = generate      ::base_t<  >  , 
               class itg_T = integrate     ::base_t<  >  ,
               class map_T = block_map_t    < prx_T   >  ,
               class com_T = comm::comm_t   < pol_T   >  >
    void enroll(  )
    {
        enroll_dat< typename prx_T::bdt_t > (    );
        block_yield = [ & ]
        ( mesh::block_t & b, mesh::mesh_t & mesh )
        {
            auto p = block_yield_base ( b , mesh );
            auto & b_h   =  prx_T::ref( p->h(  ) );
            auto & b_d   =  prx_T::ref( p->d(  ) );
            b_h.rad.n_fld(  ) = n_hv;
            b_d.rad.n_fld(  ) = n_hv;
            b_h.rad.order(  ) = space_order(  );
            b_d.rad.order(  ) = space_order(  );
            return p ;
        };
        return particle::base_t::enroll
        < prx_T, pol_T, gen_T, itg_T, map_T, com_T > (  );
    }
};
};                       // namespace particle::radiation
