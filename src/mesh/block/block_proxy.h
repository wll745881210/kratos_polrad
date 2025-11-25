#pragma once

#include "dual_data.h"

namespace mesh::block
{
////////////////////////////////////////////////////////////
// Default block proxyies: Garbage in, garbage out

template < class bdt_T >
struct proxy_t
{
    ////////// Types //////////
    using bdt_t = bdt_T;
    
    ////////// For setups //////////
    template < class bd0_T >
    static bdt_T   & ref ( const bd0_T & bd )
    {
        return dynamic_cast< bdt_T & >
               ( const_cast< bd0_T & > ( bd ) );
    };
    template < class bd0_T >
    static bdt_T   & d_i ( const bd0_T & bd )
    {
        return ref( bd );
    };  // Data block for initial setups
    
    ////////// For device data packup //////////
    template < class bd0_T >
    static bdt_T   & d   ( const bd0_T & bd )
    {
        return ref( bd );
    };  // Return ref by default, but not necessary!

    ////////// For module packing //////////
    template < class mod_T >
    static void set_module ( mod_T & mod )
    {
        return;                 // Do nothing by default
    };
};

};                              // namespace mesh::block
