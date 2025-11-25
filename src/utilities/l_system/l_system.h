#pragma once

#include "../../types.h"
#include "../types/quaternion.h"

#include <map>
#include <list>
#include <string>
#include <functional>

namespace utils
{
namespace l_system
{
////////////////////////////////////////////////////////////
// Types

struct node_t
{
    char          a;         // Status
    int           l;         // Level
    type::idx_t   x;         // Location
    quat_t< int > q;         // Quaternion for the state
    node_t(  ) : x( { 0, 0, 0 } ), a( '\0' ), l( 0 ) {  };
};

////////////////////////////////////////////////////////////
// L-system for fractal curves

class l_sys_t
{
    ////////// Con-/destructor and initializer //////////
public:
    l_sys_t(  );

    ////////// Constants and rules //////////
protected:                      // Type
    using f_op_t = std::function < void( node_t & ) >;
protected:                      // Data
    int                           l_lim;
    std::map< char, f_op_t      >  cons; // "Constants"
public:                         // Data    
    std::map< char, std::string >  rule;
public:                         // Functions
    virtual void step( node_t & n );

    ////////// Refinement //////////
protected:                      // Data
    bool                        regularize_ref;
public:                         // Data        
    std::function< bool( node_t & ) > rule_ref;
protected:                      // Functions
    virtual void refine_attempt ( node_t & n );

    ////////// Interfaces //////////
public:                         // Data
    std::list< node_t > nodes;
public:                         // Interfaces
    virtual void generate( const int & lvl );
};

};                              // namespace l_system
};                              // namespace utils
