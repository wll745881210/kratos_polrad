#include "../../utilities/string.h"
#include "../../comm/comm.h"
#include "../binary/binary_io.h"
#include "input.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

////////////////////////////////////////////////////////////
// Initializer

input::input(  ) : input_file_name ( "par.par" ),
                   is_silent( true ), merge_count( 0 )
{
    return;
}

input::input( const std::string & file_name,
              const bool & silent ) : merge_count( 0 )
{
    set_file( file_name, silent );
    return;
}

input::~input(  )
{
    return;
}

void input::set_file( const std::string & file_name,
                      const bool        &    silent )
{
    this->input_file_name = file_name;
    this->is_silent       =    silent;
    return;
}

void input::merge( const input & src )
{
    set( "args_merge", "source_" + std::to_string
       ( ++ merge_count ), src.input_file_name );
    item_map.insert( src.item_map.begin(  ) ,
                     src.item_map.end  (  ) );
    return;
}

////////////////////////////////////////////////////////////
// Communication

void input::set_comm( std::shared_ptr< comm::base_t > p )
{
    p_com = p;
    return;
}

////////////////////////////////////////////////////////////
// Read from file

std::string input::extract_prefix( const std::string & src )
{
    auto i_e =  src.find    ( ']' );
    if(  i_e == std::string::npos )
    {
        std::cerr << "Incorrect input section.\n";
        throw std::runtime_error( "input.cpp" );
    }
    return src.substr( 1, i_e - 1 );
}

void input::read_ascii(  )
{
    std::cout << "Reading input parameters from ascii : "
              << input_file_name << std::endl;

    std::string item_temp, line_temp, value_temp;
    std::stringstream ss;
    std::string prefix( "" );

    std::ifstream fin( input_file_name.c_str(  ) );

    prefix_all.clear(  );
    while( ! fin.eof(  ) )
    {
        getline ( fin, line_temp );
        ss.clear(                );
        ss.str  (      line_temp );
        std::getline   ( ss, line_temp, '#' );
        line_temp = utils::string::trim( line_temp );
        if( line_temp.empty(  ) )
            continue;
        if( line_temp[ 0 ] == '[' )
        {
            prefix = extract_prefix ( line_temp );
            prefix_all.insert       (    prefix );
            continue;
        }
        ss.clear(           );
        ss.str  ( line_temp );
        std::getline( ss,  item_temp,  '='  );
        std::getline( ss, value_temp,  '='  );
        item_temp   = utils::string::trim(  item_temp ) ;
        item_map    [ key_expand ( prefix,  item_temp ) ]
                    = utils::string::trim( value_temp ) ;
    }
    return;
}

void input::read_binary(  )
{
    std::cout << "Reading input parameters from binary: "
              << input_file_name << std::endl;
    binary_io::base_t bio;
    bio.open ( input_file_name,   "r" );
    bio.load (                        );
    bio.read ( item_map, "input_args" );
    bio.close(                        );
    for( const auto  & p :   item_map )
    {
        const auto s = utils::string::split( p.first, '|' );
        if( s.size (  ) > 1 )
            prefix_all.insert ( s[ 0 ] );
    }
    return set( "file", "binary", input_file_name );
}

bool input::read_single(  )
{
    std::fstream fin( input_file_name.c_str(  ),
                      std::ios::in | std::ios::binary );
    if( ! fin )
    {
        if( ! is_silent )
            std::cerr << "Unable to open input file: "
                      << input_file_name << std::endl;
        return false;
    }
    char  x;
    fin.read ( & x, 1 );
    fin.close(        );

    const bool is_ascii  ( 32 <= x && x <= 126 );
    is_ascii ? read_ascii(  ) :  read_binary(  );
    return true;
}

bool input::read_all(  )
{
    bool success( false );
    if( p_com->is_root(  )  )
        success = read_single(  );
    p_com->bcast( & success, sizeof( success ) );
    if( success )
    {
        p_com->bcast(   item_map );
        p_com->bcast( prefix_all );
    }
    return success;
}

bool input::read(  )
{
    if( p_com )
        return read_all(  );
    return  read_single(  );
}

////////////////////////////////////////////////////////////
// Data access

std::string input::key_expand
( const std::string & pre,  const std::string & key ) const
{
    return ( pre.empty(  ) ? "" : pre + "|" ) + key;
}

void input::msg_not_found
( const std::string & pre, const std::string & key ) const
{
    if( ! p_com || p_com->is_root(  ) )
        std::cerr << "Entry \"[" + pre + "]: "
            + key + "\" not found; using default values.\n";
    throw std::runtime_error( "input::get" );
    return;
}

bool input::found( const std::string & pre,
                   const std::string & key ) const
{
    return item_map.find( key_expand( pre, key ) ) !=
           item_map. end(  );
}

const std::map<std::string, std::string> &
input::get_item_map(  ) const
{
    return item_map;
}

const std::set<std::string> & input::get_prefixes(  ) const
{
    return prefix_all;
}
