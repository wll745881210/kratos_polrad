#pragma once

namespace mesh
{
using f_new_t  = std::function< void * ( const size_t & ) >;
using f_free_t = std::function< void   ( void         * ) >;
using f_null_t = std::function
    < void( void *,                      const size_t & ) >;
using f_cp_t   = std::function
    < void( void *, const       void *,  const size_t & ) >;

using f_read_t  = std::function
    < int ( void *,              const   std::string  & ) >;
using f_write_t = std::function
    < int ( void *, const size_t &,      const size_t & ,
                                 const   std::string  & ) >;
};                              // namespace mesh
