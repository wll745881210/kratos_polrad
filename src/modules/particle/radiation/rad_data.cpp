#include "rad_data.h"

namespace particle::radiation
{
////////////////////////////////////////////////////////////
// Basic data holder for radiation

__host__ std::string rad_t::prefix(  ) const
{
    return "rad_flx_";
}

__host__ void rad_t::setup
( const mesh::f_new_t & f_n, const mesh::geo_t & geo )
{
    auto n_gh( type::idx_t::null(  ) );
    const auto order = this->order(  );
    for( int a = 0; a < n_gh.size(  ); ++ a )
        n_gh[ a ] = ( geo.n_ceff[ a ] > 1 ? order : 0 );
    
    const auto & zeros = type::idx_t::null(  );
    mfp_i.init( f_n, geo.n_ceff,  n_gh, n_fld(  ) );
    flx  .init( f_n, geo.n_ceff, zeros, n_fld(  ) );
    return;
}

__host__ void rad_t::free( const mesh::f_free_t & f_free )
{
    mfp_i     .free( f_free );
    return flx.free( f_free );
}

__host__  void rad_t::copy_to
( const mesh::f_cp_t & f_cp, rad_t & tgt ) const
{
    mfp_i     .cp_to( tgt.mfp_i, f_cp );
    return flx.cp_to( tgt.flx,   f_cp );
}

__host__ void rad_t::read ( const mesh::f_read_t & f_r )
{
    return;
}

__host__ void rad_t::write( const mesh::f_write_t & f_w )
{
    mfp_i     .write( f_w, prefix(  ) + "mfpi_" );    
    return flx.write( f_w, prefix(  ) + "flux_" );
}

};                        // namespace particle::radiation
