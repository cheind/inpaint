/**
   This file is part of Inpaint.

   Copyright Christoph Heindl 2014

   Inpaint is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Inpaint is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with Inpaint.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>

#ifndef INPAINT_STATS_H
#define INPAINT_STATS_H

namespace Inpaint {

    /** Minimum of two */
    template<class T>
    T minimum(T a, T b)
    {
        return (std::min<T>)(a, b);
    }

    /** Maximum of two */
    template<class T>
    T maximum(T a, T b)
    {
        return (std::max<T>)(a, b);
    }

    /** Minimum of three */
    template<class T>
    T minimum(T a, T b, T c)
    {
        return (std::min<T>)(a, std::min<T>(b, c));
    }

    /** Maximum of three */
    template<class T>
    T maximum(T a, T b, T c)
    {
        return (std::max<T>)(a, std::max<T>(b, c));
    }

    template<class T>
    T clamp(T x, T inclusiveMin, T inclusiveMax)
    {
        return maximum(inclusiveMin, minimum(x, inclusiveMax));
    }

    template<class T>
    T clampLower(T x, T inclusiveMin)
    {
        return maximum(inclusiveMin, x);
    }

    template<class T>
    T clampUpper(T x, T inclusiveMax)
    {
        return minimum(inclusiveMax, x);
    }

}
#endif
