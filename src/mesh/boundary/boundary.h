#pragma once

#include "../../device/device.h"
#include "../../io/args/input.h"
#include "../block/dual_data.h"
#include <functional>
#include <memory>
#include <atomic>
#include <future>

namespace mesh
{
struct mod_base_t;
};

namespace mesh::boundary
{
////////////////////////////////////////////////////////////
// Generic kernel template for boundaries.

template< class bdt_T, class bnd_T > __global__ void 
phy_ker ( const bdt_T bdt, const bnd_T bnd, const int step )
{
    return bnd( bdt, step );
}

////////////////////////////////////////////////////////////
// Holder for binding host actions and data

struct base_t;

struct holder_base_t
{
    std   ::shared_ptr<         base_t >      p;
    std   ::shared_ptr< device::base_t >  p_dev;
    block ::    dual_t                    * p_d;
    device::   event_t                    event;
    device::  stream_t                   stream;
};

struct holder_p_t : holder_base_t   // Physical
{
    int      side ;
    std::function
    < void( holder_p_t &, const int & ) > launch;
};

struct holder_c_t : holder_base_t  // Communication
{
    holder_c_t( block::dual_t & );
    virtual void setup( const neighbor_t &, mod_base_t & );
    bool    same_rank (  )    const;

    int          d_lvl;     int                tag_s;
    int           rank;     int                tag_r;
                            int                tag_f;

    std::shared_ptr     < comm ::base_t >  p_com;
    std::function< void ( const int & ) > f_pack; // Packing
    std::function< void ( const int & ) > f_unpk; // Unpacki
    std::function< void ( const int & ) > f_fpck; // Fluxes
    std::function< void (             ) > f_send;
    std::function< void (             ) > f_recv;
    std::function< void (             ) > f_fcom; // Flux
};

////////////////////////////////////////////////////////////
// Base struct for a boundary condition

struct base_t
{
    ////////// Data //////////
    int              n_th_lim;

    ////////// Interfaces //////////    
    __host__ virtual void init
    ( const  input &,                   mod_base_t & ) {  };
    __host__ virtual void free  ( const holder_c_t & ) {  };
    __host__ virtual void setup_p
    ( holder_p_t &,                     mod_base_t & ) {  };
    __host__ virtual void setup
    ( holder_c_t &, const neighbor_t &, mod_base_t & ) {  };
    __host__ virtual void set_geo
    ( holder_c_t &, const neighbor_t &, mod_base_t & ) {  };
    __host__ virtual void set_mem
    ( holder_c_t &, const neighbor_t &, mod_base_t & ) {  };
    __host__ virtual void set_comm
    ( holder_c_t &, const neighbor_t &, mod_base_t & ) {  };
    __host__ virtual void set_local
    ( holder_c_t &, const neighbor_t &, mod_base_t & ) {  };
    __host__ virtual void set_launch
    ( holder_c_t &, const neighbor_t &, mod_base_t & ) {  };
};

struct comm_base_t : public base_t
{
    ////////// Data //////////
    bool            same_rank;
    bool                  rhs;
    int                  mode;
    int                 d_lvl;
    int                 order;
    idx_t                axis;
    idx_t            n_ceff_s;
    idx_t            n_ceff_r;    
    idx_t            offset_s;  // Offset: send
    idx_t            offset_r;  // Offset: recv
    idx_t            offset_c;  // Offset: copy

    ////////// Interfaces //////////
    __host__ virtual std:: tuple < int, int >
    resource_thbl  ( const idx_t & n_cell ) const;
    __host__ virtual void  setup
    ( holder_c_t &, const  neighbor_t &, mod_base_t & )    ;
    __host__ virtual void  set_geo
    ( holder_c_t &, const  neighbor_t &, mod_base_t & )    ;
};

};                              // namespace mesh::boundary
