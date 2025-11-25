#include "block.h"

namespace mesh
{
////////////////////////////////////////////////////////////
// The "abstract" block type

block_t::block_t(  ) : id_g( -1 ), id_l( -1 ), rank( -1 )
{
    return;
}

bool block_t::updated_neighbor(  ) const
{
    return true;
}  // AMR interface

std::string block_t::io_prefix(  ) const
{
    return "block_" + std::to_string( id_g ) + "|";
}

void block_t::read ( binary_io::base_t & bio ,
                     device   ::base_t & dev )
{
    return geo.read(  dev.f_malloc_host  ,
    [ & ] ( void * p, const std::string    & t )
    {   return bio.read ( p, io_prefix(  ) + t ); } );
}

void block_t::write( binary_io::base_t & bio ) const
{ 
    bio.write( reg.x, 3,   io_prefix(  ) + "i_logic" );
    bio.write( reg.level,  io_prefix(  ) +   "level" );    
    return geo.write
    ( [ & ]  ( void   * p, const size_t & c, const
               size_t & u, const std::string & t )
    { return bio.write( p, c, u, io_prefix(  ) + t ); } );
}
};                              // namespace mesh
