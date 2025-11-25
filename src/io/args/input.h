#pragma once

#include "../../utilities/types/my_type_traits.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <string>
#include <vector>
#include <set>
#include <map>

////////////////////////////////////////////////////////////
// Types

template< typename T, template< typename >
          class tester_T, bool  positive = true >
using if_t =  type::if_t< T, tester_T, positive >;

namespace comm
{
struct base_t;
};

////////////////////////////////////////////////////////////
// class input

class input
{
    ////////// Initializer //////////
public:                          // Function
     input(  );
    ~input(  );
     input       ( const std::string & file_name,
                   const bool        & silent   = true );
    void set_file( const std::string & file_name,
                   const bool        & silent   = true );
    void merge   ( const input       & src );

    ////////// Comm //////////
protected:                      // Module
    std::shared_ptr< comm::base_t > p_com;
public:                         // Function
    void   set_comm( std::shared_ptr< comm::base_t > p );

    ////////// Read from parameter file //////////
protected:                       // Data
    int                               merge_count;
    std::string                   input_file_name;
    std::map< std::string, std::string > item_map;
    std::set< std::string >            prefix_all;
protected:                       // Function
    std::string extract_prefix( const std::string & src );
protected:                       // Function
    void read_ascii (  );
    void read_binary(  );
    bool read_single(  );
    bool read_all   (  );
public:                         // Functions
    bool read       (  );

    ////////// Accesses the entries //////////
protected:                       // Data
    bool is_silent;
protected:                       // Functions
    std::string key_expand( const std::string & pre,
                            const std::string & key ) const;
    void     msg_not_found( const std::string & pre,
                            const std::string & key ) const;
protected:                      // Type-trait
    void assign
    ( std::stringstream & ss, std::string & t ) const
    {
        ss >> t;
        return;
    };
    template< class T, if_t< T, std::is_fundamental > = 0 >
    void assign( std::stringstream & ss, T & t ) const
    {
        ss >> t;
        return;
    };
    template< class T, if_t< T, std::is_pointer     > = 0 >
    void assign( std::stringstream & ss, T & t ) const
    {
        typedef typename
        std::remove_pointer< decltype( t ) >::type elem_T;
        std::vector< elem_T > v;
        elem_T tmp;
        while(  ss    >>  tmp  )
            v.push_back(  tmp  );
        t = new elem_T[ v.size (  ) ];
        std::copy(   v.begin(  ), v.end(  ), t );
        return;
    };
    template< class T, if_t< T, std::is_array       > = 0 >
    void assign( std::stringstream & ss, T & t ) const
    {
        using elem_T =  std::remove_all_extents_t< T >;
        const size_t size = sizeof( t ) / sizeof( elem_T );
        for(  size_t i = 0; i < size; ++ i )
            ss >> t[ i ];
        return;
    };
    template< class T, if_t< T, type::is_std_array  > = 0 >
    void assign( std::stringstream & ss, T & t ) const
    {
        for( size_t i = 0; i < t.size(  ); ++ i )
            ss >> t[ i ];
        return;
    };
    template< class T, if_t< T, type::is_vec_3d     > = 0 >
    void assign( std::stringstream & ss, T & t ) const
    {
        for( size_t i = 0; i < 3; ++ i )
            ss >> t[ i ];
        return;
    };
    template< class T, if_t< T, type::is_seq        > = 0 >
    void assign( std::stringstream & ss, T & t ) const
    {
        t.clear(  );
        typename T::value_type tmp;
        while( ss >> tmp )
            t.push_back( tmp );
        return;
    };
    template < class T >   // Do not compile if all missed
    void assign( T & t, ... ) const = delete;
public:                         // Functions
    template< typename T > bool operator (  )
    ( const std::string & pre,
      const std::string & key, T & res ) const
    {
        const auto p = item_map.find
                     ( key_expand( pre, key ) );
        const bool item_found( p != item_map.end(  ) );
        std::stringstream ss;
        if( item_found )
        {
            ss.str( p->second  );
            assign( ss,   res  );
        }
        return item_found;
    };
    template< typename T >
    T get( const std::string & pre,
           const std::string & key ) const
    {
        T res;
        if( ! ( ( * this ) ( pre, key, res ) ) )
            msg_not_found( pre, key );
        return res;
    };
    template< typename T, typename D >
    T get( const std::string & pre, const std::string & key,
           const D   & default_v  ) const
    {
        T res;
        if( ! ( ( * this ) ( pre, key, res ) ) )
            res = default_v;
        return res;
    };
    template< typename T, typename D >
    T get_set
    ( const std::string & pre, const std::string & key,
      const D   & default_v  ) const
    {
        T res;
        if( ! ( ( * this ) ( pre, key, res ) ) )
        {
            res = default_v;
            const_cast< input * >
                      ( this ) ->set( pre, key, res );
        }
        return res;        
    };
    
    template< typename T >
    void set( const std::string & pre,
              const std::string & key, const T & val )
    {
        std::stringstream ss;
        ss << val;
        item_map[ key_expand ( pre, key ) ] = ss.str(  );
        if( ! pre.empty(  ) )
            prefix_all.insert( pre );
        return;
    };
    bool found( const std::string & pre,
                const std::string & key ) const;
public:                         // Overall map interfaces
    const std::map< std::string, std::string > &
    get_item_map(  ) const;
    const std::set< std::string> & get_prefixes(  ) const;
};
