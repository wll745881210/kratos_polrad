#include <iostream>
#include "user/probgen.h"

////////////////////////////////////////////////////////////
// Main function

int main( int argc, char * argv[  ] )
{
    if( argc <= 1 )
        throw std::runtime_error( "No input file." );
    prob::run( argc, argv );
    return  0;
}
