#pragma once

#include "../io/args/input.h"
#include "../comm/comm.h"

#include <memory>

////////////////////////////////////////////////////////////
// User interface

namespace prob
{
void run( int argc, char * argv[  ] ) ;
void cmd( int argc, char * argv[  ] , input & args );
};                               // namespace prob
