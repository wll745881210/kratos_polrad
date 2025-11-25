#pragma once

#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

#include "../comm/comm.h"
#include "../device/device.h"
#include "../io/args/input.h"
#include "../io/binary/binary_io.h"

#include "block/block.h"
#include "block/block_proxy.h"
#include "cycle/cycle.h"
#include "boundary/boundary.h"
#include "meshgen/meshgen.h"
#include "module_base/module_base.h"
#include "tree/tree.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// Mesh keeper: The foundation of everything

class mesh_t
{
    ////////// Constructor //////////
public:
    mesh_t(  );
   ~mesh_t(  );

    ////////// Initialization //////////
protected:                      // Data
    bool                                    restart;
    int                                   order_max;
    std::map< std::string, std::string > input_args;
protected:                      // Modules
    std::vector<       std:: function
               < void( const input  & ) > > reads;
    std::map   < int,  std:: function
               < void( const input  & ) > > inits;
public:                         // Function
    virtual void  init ( const input  &  args );
    virtual const bool & is_restart(  ) const  ;
    virtual bool         changed   (  ) const  ;
    virtual const int  & n_order   (  ) const  ;

    ////////// Binary IO //////////
public:                         // Function
    virtual void save( const std::string & file_name );

    ////////// Task cycle //////////
protected:                      // Modules
    std::map< int, std::function< void(  ) > > steps;
public:                         // Function
    virtual void   step(  );
    virtual void evolve(  );

    ////////// Block structures and access //////////
protected:                      // Types
    using p_blk_t   = std::shared_ptr  < block_t >;
protected:                      // Data
    lex_map_t< region_logic_t, p_blk_t > block_map;
    std::shared_ptr           < tree_t >     p_tre;
public:                         // Interface
    tree_t &          tree(  )         const;
    std::function< p_blk_t(  ) > f_blk_yield;
public:                         // Creation/destruction
    virtual void      remove  ( const region_logic_t & );
    virtual bool      contains( const region_logic_t & );
    virtual block_t & emplace ( const region_logic_t & );
    virtual block_t & block   ( const region_logic_t & );
    virtual block_t & front(  );
    virtual block_t & back (  );
public:                         // Interface
    template< class tre_T >
    void enroll_tree (  ) ;
    template< class blk_T >
    void enroll_block(  ) ;

    ////////// Range-based for loop over locals //////////
protected:                      // Modules
    using block_map_t =lex_map_t< region_logic_t, p_blk_t >;
    struct iter_t : public block_map_t::iterator
    {
        iter_t( const block_map_t::iterator & it );
        block_t     &              operator * (  );
    };
public:                         // Functions
    iter_t begin(  );
    iter_t   end(  );

    ////////// Fundamental modules //////////
public:                         // Modules
    std::shared_ptr< comm     ::base_t > p_com;
    std::shared_ptr< cycle    ::base_t > p_cyc;
    std::shared_ptr< device   ::base_t > p_dev;
    std::shared_ptr< meshgen  ::base_t > p_mgn;    
    std::shared_ptr< binary_io::base_t > p_bio;
public:                         // Functions
    template < class mgn_T >
    std::shared_ptr< mgn_T > enroll_mgen     (  );    
    template < class com_T >
    std::shared_ptr< com_T > enroll_comm     (  );
    template < class cyc_T >
    std::shared_ptr< cyc_T > enroll_cycle    (  );
    template < class dev_T >
    std::shared_ptr< dev_T > enroll_device   (  );
    template < class bio_T >
    std::shared_ptr< bio_T > enroll_binary_io(  );

    ////////// Physical Modules //////////
protected:                      // Modules for physics
    std::vector< std::shared_ptr< mod_base_t > > mods;
public:                         // Functions for enrolling
    template< class mod_T >  std::shared_ptr< mod_T >
    enroll_module( const int & i_i, const int & i_s );
    template< class mod_T >  std::shared_ptr< mod_T >
    enroll_module(                                  );
};                              // class     mesh_t

};                              // namespace mesh
