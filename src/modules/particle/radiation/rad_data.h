#pragma once

#include "../types/block_data.h"

namespace particle::radiation
{
////////////////////////////////////////////////////////////
// Basic data holder for radiation

struct  rad_t
{
    ////////// Types //////////
    using f_t = type::float_t;

    ////////// Data //////////
    mesh::dat_3d_t< f_t > mfp_i;
    mesh::dat_3d_t< f_t >   flx;

    ////////// Functions //////////
    __host__ __device__ __forceinline__ int & order(  )
    {
        return mfp_i.n_gh[ 0 ];
    };    
    __host__ __device__ __forceinline__
    const    int & n_fld(  ) const
    {
        return flx.n_int;
    };
    __host__ int & n_fld(  )
    {
        return flx.n_int;
    };

    __host__ virtual std::string prefix(    )    const ;
    __host__ virtual void setup( const mesh::  f_new_t & ,
                                 const mesh::    geo_t & );
    __host__ virtual void free ( const mesh:: f_free_t & );
    __host__ virtual void read ( const mesh:: f_read_t & );
    __host__ virtual void write( const mesh::f_write_t & );
    __host__ virtual void copy_to
    ( const mesh::f_cp_t & f_c, rad_t & tgt ) const;
};
};                        // namespace particle::radiation
