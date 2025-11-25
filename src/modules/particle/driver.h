#pragma once

#include "../../mesh/mesh.h"
#include "types/driver_base.h"

#include <functional>
#include <memory>
#include <vector>

namespace particle
{
////////////////////////////////////////////////////////////
// Particle evolution

class base_t : public mesh::mod_base_t
{
    ////////// Interfaces //////////
public:                         // Functions
    virtual void init       ( const input  & args ,
                              mesh::mesh_t & mesh );
    virtual void finalize   ( mesh::mesh_t & mesh );
    virtual void step       ( mesh::mesh_t & mesh );
    virtual void save       ( mesh::mesh_t & mesh,
                              binary_io::base_t & );
    virtual int  space_order(                     );
    using   mesh::mod_base_t::update;
    virtual void update
    ( const mod_base_t::v_reg_t &, mesh::mesh_t & );
    virtual comm_mode_t  comm_mode(               );

    ////////// Modules //////////
public:                         // Data
    int                               n_cyc;
    device:: event_t                  event;    
    device::stream_t                 stream;
    std::shared_ptr< driver::base_t > p_pol;
    std::shared_ptr< driver::base_t > p_itg;
    std::shared_ptr< driver::base_t > p_gen;
    std::shared_ptr< driver::base_t > p_map;
    std::shared_ptr< driver::base_t > p_pcm;
    std::vector< std::shared_ptr< driver::base_t > > drv;
protected:                      // Callable
    std::function
    < void( mesh::mesh_t &, const int & ) >    integrate;
    std::function< void
    ( mesh::mesh_t &, binary_io::base_t & ) > f_par_save;
public:                         // Functions
    template< class prx_T, class pol_T, class gen_T ,
              class itg_T, class map_T, class com_T >
    void enroll(  )
    {
        p_pol = std::make_shared< pol_T > (  );
        p_gen = std::make_shared< gen_T > (  );
        p_itg = std::make_shared< itg_T > (  );
        p_map = std::make_shared< map_T > (  );
        p_pcm = std::make_shared< com_T > (  );
        drv   = {  p_pol, p_itg,  p_gen, p_map, p_pcm };

        integrate =  [ & ]
        ( mesh::mesh_t & mesh, const int & step )
        {
            auto & pol = dynamic_cast< pol_T & >( * p_pol );
            auto & gen = dynamic_cast< gen_T & >( * p_gen );
            auto & itg = dynamic_cast< itg_T & >( * p_itg );
            auto & map = dynamic_cast< map_T & >( * p_map );
            auto & com = dynamic_cast< com_T & >( * p_pcm );
            this-> sync_streams( mesh );
            map.set_dt   (                 * this ) ;
            itg.pre_proc ( pol, map, com,  * this ) ;
            gen.generate ( pol, map,       * this ) ;
            do
            {
                com.prep ( pol,            * this ) ;
                itg.intg ( pol, map, com,  * this ) ;
            }   while    ( com.proc( pol,  * this ) );
            itg.post_proc( pol, map, com,  * this ) ;
            p_bdk->act_bnd         ( mesh,   step ) ;
            this->sync_streams     ( mesh         ) ;
        };
        f_par_save = [ & ] ( mesh     ::mesh_t & mesh,
                             binary_io::base_t & bio )
        {
            const std::   string  prefix = "par_rank_"
                + std::to_string( mesh.p_com->rank(  ) );

            dynamic_cast< pol_T & >( * p_pol ).write
            (   p_dev->f_cp,  [ & ]
              ( void          * p, const      size_t & c ,
                const size_t  & u, const std::string & t )
            {
                return bio.write
                     ( p, c, u, prefix + t, true );
            }   );
        }   ;
        if( ! f_bdt_yield )
            enroll_dat< typename prx_T::bdt_t > (  );
        if( ! block_yield )
            throw  std::runtime_error
                ( "prx_T not enrolled for particle" );
        return prx_T::set_module( * this );
    };
};
};                              // namespace parteicle
