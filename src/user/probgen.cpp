#include "probgen.h"
#include <iostream>

namespace prob
{
////////////////////////////////////////////////////////////
// Default probgen (do nothing!)
void cmd( int argc, char * argv[  ], input & args )
{
    for( int i = 1; i < argc; ++ i )
    {
        args.set_file(  argv [ i ] );
        args.read    (             );
    }   // Overriding the previous args with the current
    return;
}

void __attribute__((weak)) run( int argc, char * argv[  ] )
{
    std::cerr << "Default run function is empty!\n";
    return;
}

};                              // namespace prob
