#pragma once

#include <sstream>
#include <vector>

namespace utils::string
{
////////////////////////////////////////////////////////////
// Utility functions for strings

inline std::string trim_right
( std::string s, const std::string delim = " \f\n\r\t\v" )
{
    return s.erase( s.find_last_not_of( delim ) + 1 );
}

inline std::string trim_left
( std::string s, const std::string delim = " \f\n\r\t\v" )
{
    return s.erase( 0, s.find_first_not_of( delim ) );
}

inline std::string trim
( std::string s, const std::string delim = " \f\n\r\t\v" )
{
    return trim_left( trim_right( s, delim ), delim );
}

inline void trim_space( std::string & str )
{
    if( str.empty(  ) )
        return;
    const size_t first = str.find_first_not_of( ' ' );
    const size_t  last = str.find_last_not_of ( ' ' );
    str   = str.substr( first, ( last - first + 1 ) );
    return;
}

inline  std::vector < std::string > split
( const std::string & src, const char & delim = ' ' )
{
    std::istringstream ss( src );
    std::string token;
    std::vector< std::string > result;

    while( std::getline( ss, token, delim ) )
        result.push_back( token );
    return result;
}

inline  int    replace_all
( std:: string & str, const std::string & search,
  const std::string & replace )
{
    if( replace.find( search, 0 ) != std::string:: npos )
        throw std::runtime_error( "string: illegal rep" );
    size_t pos( 0 );
    int  count( 0 );
    while( true )
    {
        pos = str.find( search,  pos );
        if( pos == std::string::npos )
            break;
        str.erase ( pos, search.length(  ) );
        str.insert( pos, replace );
        pos += replace.length(   );
        ++ count;
    }
    return count;
}

};                              // namespace utils::string
