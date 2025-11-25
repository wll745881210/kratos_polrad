#include "../boundary/boundary.h"
#include "dual_data.h"

namespace mesh::block
{
////////////////////////////////////////////////////////////
// Constructor and destructor

void dual_t::set_dev
( std::shared_ptr< device::base_t > p_dev )
{
    this  -> p_dev = p_dev;
    event  = p_dev ->event_create(  );
    stream = p_dev ->yield_stream(  );
    return;
}

void dual_t::free(  )
{
    if( p_d )
    {
        p_d->free    ( p_dev->f_free   );
        p_d->geo.free( p_dev->f_free   );
    }
    if( p_h )
        p_h->free ( p_dev->f_free_host );
    return p_dev->event_destroy( event );
}

void dual_t::setup(  )
{
    p_d->geo.setup( b_w.get(  ).geo, p_dev-> f_malloc );
    b_w.get(  ).geo .  cp_shlw( p_h->geo              );
    b_w.get(  ).geo .  cp_deep( p_d->geo, p_dev->f_cp );
    // ^--- Set mem for geometry \\ v-- Set mem for data
    p_h->setup( p_dev->f_malloc_host );
    p_d->setup( p_dev->f_malloc      );
    return;
}

////////////////////////////////////////////////////////////
// Data transfer and IO

void dual_t::copy_h2d(  )
{
    return p_h->copy_input ( p_dev->f_cp, * p_d );
};

void dual_t::copy_d2h(  )
{
    return p_d->copy_output( p_dev->f_cp, * p_h );
};

std::string dual_t:: prefix(  )
{
    return b_w.get(  ).io_prefix(  );
}

void dual_t:: read( binary_io::base_t & bio )
{
    p_h->read( [ & ]( void * p, const std::string & t )
    {    return bio.read( p, prefix(  ) + t );    }   );
    return;
}

void dual_t::write( binary_io::base_t & bio ,
                    const int & output_flag )
{
    if( output_flag <= 0 )
        return;
    this->copy_d2h(  );
    h(  ).output_flag = output_flag;
    p_h->write( [ & ]( void   * p, const size_t & c, const
                       size_t & u, const std::string & t )
    {    return bio.write( p, c, u, prefix(  ) + t );  } );
    return;
}
};                              // namespace mesh::block
