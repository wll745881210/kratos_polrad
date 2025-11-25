#pragma once

#include <sstream>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////
// String operations

namespace str_op
{

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

inline int replace_all
( std::string & s, const std::string & pat,
                   const std::string & rep )
{
    size_t pos = 0;
    int  count = 0;
    while( true )
    {
        pos = s.find( pat,  pos );
        if( pos == std::string::npos )
            break;
        s.erase ( pos, pat.length(  ) );
        s.insert( pos, rep            );
        pos += rep.length(  );
        ++  count;
    }
    return count;
}

template< typename S >
std::vector< std::string > split_string
( const S & s, const char delim )
{
    std::istringstream ss( s );
    std::string token;
    std::vector< std::string > result;
    while( std::getline( ss, token, delim ) )
        result.push_back( token );
    return result;
}

};                              // namespace str_op


