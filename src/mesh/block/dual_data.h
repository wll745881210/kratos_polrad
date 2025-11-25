#pragma once

#include "block.h"
#include "block_data.h"

namespace mesh::block
{
////////////////////////////////////////////////////////////
// Block data on both host and device sides

struct dual_t
{
    ////////// Data //////////
    int                                order;
    std::shared_ptr<     base_data_t >   p_d; // Device
    std::shared_ptr<     base_data_t >   p_h; // Host
    std::reference_wrapper < block_t >   b_w; // Block

    ////////// Device and load balancing //////////
    std::shared_ptr< device:: base_t > p_dev;
    device:: event_t                   event;
    device::stream_t                  stream;

    ////////// Data access //////////
    inline   block::base_data_t & d  (  ) const
    { return const_cast< block::base_data_t & >( * p_d ); };
    inline   block::base_data_t & h  (  ) const
    { return const_cast< block::base_data_t & >( * p_h ); };
    inline   const        geo_t & geo(  ) const
    { return                      b_w.get (   ).   geo  ; };

    ////////// Con-/destructor //////////
    dual_t( block_t & b ) : b_w( b ) {  };
    virtual void set_dev
    ( std::shared_ptr< device::base_t > p_dev );
    virtual void free (  );
    virtual void setup(  );

    ////////// Data transfer and IO //////////
    virtual void copy_h2d     (  );
    virtual void copy_d2h     (  );
    virtual std::string prefix(  );
    virtual void read ( binary_io::base_t & );
    virtual void write( binary_io::base_t &, const int & );
};
};                              // namespace mesh::block
