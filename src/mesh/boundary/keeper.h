#pragma once

#include "../../io/args/input.h"
#include "../../comm/comm.h"
#include "../../device/device.h"
#include "boundary.h"
#include <map>
#include <array>
#include <vector>
#include <string>
#include <memory>
#include <functional>

namespace mesh
{
class         mesh_t;
struct    mod_base_t;

namespace   boundary
{
////////////////////////////////////////////////////////////
// Boundary condition keeper

class  keeper_t
{
    ////////// Initializer //////////
protected:                      // Data
    int       n_th_lim;
protected:                      // Modules
    std::vector< std::function
               < void( const input & ) > > inits;
public:                         // Functions
    keeper_t(  ) ;
    virtual void init( const input & , mesh_t & );

    ////////// Boundary assignment //////////
protected:                      // Data
    bool  independent_stream;
protected:                      // Functions
    virtual void assign_strm ( holder_base_t     & );
public:                         // Functions
    virtual void assign_comm ( block_t &, mesh_t & );
    virtual void assign_phys ( block_t &, mesh_t & );
    virtual bool desired_mode( const  neighbor_t & );
    
    ////////// Modules //////////
public:                         // Modules
    std::shared_ptr< device::base_t > p_dev;
    std::shared_ptr<   comm::base_t > p_com;
    mod_base_t                      * p_mod;    

    ////////// Physical boundary //////////
public:                         // Modules
    std::map< std::string, holder_p_t >        holder_phys;
    std::vector          < holder_p_t >          pool_phys;
    lex_map_t
    < region_logic_t, std::vector< holder_p_t > > bnd_phys;
public:                         // Functions
    template  < class bnd_T >
    std::shared_ptr < bnd_T >
    enroll_phys( const std::string & tag );

    ////////// Communication boundary //////////
protected:                      // Type
    using h_map_t = lex_map_t< neighbor_t, holder_c_t >;
public:                         // Data
    lex_map_t < region_logic_t, h_map_t >      bnd_comm;
public:                         // Closures
    std::function< void( holder_c_t & ) >  f_comm_yield;
public:                         // Functions
    template< class com_T >
    void enroll_comm(   ) ;

    ////////// Interfaces //////////
public:                         // Function
    virtual void update ( mesh_t & ) ;
    virtual void act_bnd( mesh_t & , const  int & ) {  };
    virtual void act_flx( mesh_t & , const  int & ) {  };
};

////////////////////////////////////////////////////////////
// Enrolling physical boundary

template < class bnd_T >
std::shared_ptr< bnd_T > keeper_t::enroll_phys
( const std::string & tag )
{
    auto   p = std::make_shared < bnd_T >(  );
    auto & h = holder_phys[ tag ];
    h.p      = p;
    h.launch = [ & ,  p ] 
    ( holder_p_t & h, const int & step )
    {
        auto & d( * h.p_d );
        p -> side = h.side ;
        p_dev->event_wait  ( h.stream, d. event );
        p->launch          ( h,  step, *  p_dev );
        p_dev->event_record( h. event, h.stream );
        p_dev->event_wait  ( d.stream, h. event );
    }; 
    inits.push_back( [ = ] ( const input & args )
    {      p ->init( args, * p_mod );    }      );
    return p;
}


////////////////////////////////////////////////////////////
// Enrolling communication boundary

template< class com_T >
void keeper_t:: enroll_comm(  )
{
    f_comm_yield = [ & ] ( holder_c_t & h )
    {
        auto   p = std::make_shared< com_T > (   );
        p->n_th_lim = n_th_lim;
        h.p_dev     =    p_dev;
        h.p_com     =    p_com;
        h.p         =        p;
    };
    return;
}

};
};                              // namespace boundary::mesh
