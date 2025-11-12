#pragma once

#include <vector>

namespace aks
{

template<typename T, typename F>
void unordered_erase(std::vector<T>& xs, F predicate)
{
    if(xs.empty()){
        return;
    }

    auto pop_back = []( auto& ys ){
        auto y = std::move( ys.back() );
        ys.pop_back();
        return y;
    };

    for(size_t idx = 0; idx < xs.size();)    
        if( predicate( xs[ idx ] ) )
            xs[ idx ] = pop_back( xs );
        else             
            ++idx;    
}

template<typename T, typename F>
void unordered_erase(std::pmr::vector<T>& xs, F predicate)
{
    if(xs.empty()){
        return;
    }

    auto pop_back = []( auto& ys ){
        auto y = std::move( ys.back() );
        ys.pop_back();
        return y;
    };

    for(size_t idx = 0; idx < xs.size();)    
        if( predicate( xs[ idx ] ) )
            xs[ idx ] = pop_back( xs );
        else             
            ++idx;    
}

}
