#pragma once

#include "../../../mesh/mesh.h"

namespace particle
{
////////////////////////////////////////////////////////////
// Block data type with 

template< class bdt_T , class  dat_T >
struct   block_data_t : public bdt_T
{
    ////////// Type and data //////////
    using super_t = bdt_T;
    dat_T             dat;

    ////////// Functions //////////
    __host__ virtual void setup_mem
    ( const mesh::f_new_t & f_n )
    {
        super_t::  setup_mem       ( f_n );
        return dat.setup( f_n, this->geo );
    };

    __host__ virtual void setup_mem
    ( const mesh::f_read_t & f_r ,
      const mesh:: f_new_t & f_n )
    {
        super_t::setup_mem( f_r,       f_n );
        return   dat.setup( f_n, this->geo );
    };

    __host__ virtual void free( const mesh::f_free_t & f_f )
    {
        super_t :: free( f_f );
        return dat.free( f_f );
    };

    __host__ virtual void copy_to
    ( const mesh::f_cp_t & f_cp, mesh::block::base_data_t &
      tgt_, const bool & cp_geo )      const
    {
        auto & tgt( tgt_.derive< block_data_t >(  ) );
        super_t :: copy_to( f_cp, tgt_, cp_geo )  ;
        return dat.copy_to( tgt.dat,      f_cp )  ;
    };
    __host__ virtual void write
    ( const mesh::f_write_t & f_w )
    {
        super_t :: write( f_w );
        return dat.write( f_w );
    };
};
};                              // namespace particle
