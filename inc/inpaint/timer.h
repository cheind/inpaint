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

#ifndef INPAINT_TIMER_H
#define INPAINT_TIMER_H

#include <opencv2/core/utility.hpp>

namespace Inpaint {

    /** 
        Simple timer for profiling based on OpenCV functionality.        
    */
    class Timer {
    public:
        inline Timer()
          : _invFreq(1.0 / cv::getTickFrequency()), _start(cv::getTickCount())
        {}

        inline double measure()
        {
            int64 stop = cv::getTickCount();
            double elapsed = (stop - _start) * _invFreq;
            _start = stop;
            return elapsed;
        }

    private:
        int64 _start;
        double _invFreq;
    };

    class TimerWithStats {
    public:

        inline void measure(int index) 
        {
            _stats[index].sum += _t.measure();
            _stats[index].called += 1;
        }

        inline double mean(int index) const
        {
            return _stats[index].sum / _stats[index].called;
        }

        inline double total(int index) const
        {
            return _stats[index].sum ;
        }

    private:

        struct Stats {
            int64 called;
            double sum;

            Stats()
                : called(0), sum(0)
            {}
        };

        Stats _stats[10];
        Timer _t;
    };

}
#endif