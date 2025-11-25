#pragma once

#include "../../io/args/input.h"
#include "../../io/binary/binary_io.h"
#include "../block/dual_data.h"
#include "../boundary/keeper.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Default shape of modules

class      mesh_t;

struct mod_base_t
     : public std::enable_shared_from_this< mod_base_t >
{
    ////////// Types //////////
    using v_reg_t = std::vector      <     region_logic_t >;
    using p_bdt_t = std::shared_ptr  < block::base_data_t >;
    using p_ddt_t = std::shared_ptr  < block::     dual_t >;
    using b_map_t = lex_map_t   < region_logic_t, p_ddt_t >;
    using f_blk_t = std::function
        < void( const region_logic_t &, block::dual_t & ) >;

    std::shared_ptr <  mod_base_t > // Dangerous, don't use
    shared_from_this(  ) = delete ; // Weak pointer only

    ////////// Data access //////////
    bool                                 need_update;
    int                                  output_flag;
    std::shared_ptr < b_map_t >          p_blk_local;
    p_ddt_t       & p_data( const region_logic_t & );
    block::dual_t &   data( const region_logic_t & );
    void          map_mesh( const        f_blk_t & );

    ////////// Range-based for loop //////////
    struct iter_t : public b_map_t::iterator
    {
        iter_t( const b_map_t::iterator & it );
        block::dual_t &        operator * (  );
    };
    iter_t begin(  );
    iter_t   end(  );

    ////////// Module ////////// v--ptr to parasitised mod
    std::  weak_ptr<            mod_base_t   >       q_mod;
    std::shared_ptr< device    ::   base_t   >       p_dev;
    std::shared_ptr< boundary  :: keeper_t   >       p_bdk;
    std::function  < p_bdt_t   ( block_t & ) > f_bdt_yield;
    std::function  < p_ddt_t   ( block_t & ) > f_ddt_yield;
    std::function  < p_ddt_t   ( block_t & ,
                                  mesh_t & ) > block_yield;
    mesh_t      *                                   p_mesh;

    ////////// Functions //////////
    mod_base_t(  )  ;
    virtual void read  ( const input & )             {  };
    virtual void init  ( const input & ,  mesh_t & )     ;
    virtual void step                  (  mesh_t & ) {  };
    virtual void sync_streams          (  mesh_t & )     ;
    virtual void evolve                (  mesh_t & ) {  };
    virtual void finalize              (  mesh_t & )     ;

    virtual void update                (  mesh_t & )     ;
    virtual void update( const v_reg_t &, mesh_t & )     ;
    virtual void update
    ( const v_reg_t   &, const v_reg_t &, mesh_t & ) {  };
    virtual void init_cond     ( block :: dual_t & ) {  };
    virtual void parasite( std::weak_ptr< mod_base_t >  );

    ////////// Interfaces //////////
    virtual int     space_order     (   ) {  return 0   ; };
    virtual p_ddt_t block_yield_base( block_t &, mesh_t & );
    virtual void    save  ( mesh_t &, binary_io::base_t & );

    template < class bdk_T >
    std::shared_ptr< bdk_T > enroll_bdk (   )
    {
        auto    q = std::make_shared< bdk_T > (  );
        q ->p_mod = this;
        p_bdk     =    q;
        return         q;
    };
    template< class bdt_T, class ddt_T = block::dual_t >
    void enroll_dat (  )
    {
        f_bdt_yield = [ & ]  ( block_t & b )
        {
            return std::make_shared< bdt_T > (   );
        };
        f_ddt_yield = [ & ]  ( block_t & b )
        {
            return std::make_shared< ddt_T > ( b );
        };
    };
};
};                              // namespace mesh
