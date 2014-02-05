#http://scikit-image.org/
#source taken from above

import numpy as np
import rpy2.robjects as robjects
import random
import math as pm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pylab as p
import mpl_toolkits.mplot3d.axes3d as p3
import math

#import gaussian_fit
import sum_of_gaussians_interpolation as sog
from netcdf_reader import *
#from spline_cdf_curve_morphing import *
from mayavi.mlab import *
from peakfinder import *
from quantile_lerp import *

from collections import deque

_param_options = ('high', 'low')

MEM = 600

def _get_fraction(from_value, to_value, level):
    if to_value == from_value:
        return 0
    return ((level - from_value) / (to_value - from_value))
    
def iterate_and_store(array, level, vertex_connect_high, interp='crisp'):
    """Iterate across the given array in a marching-squares fashion,
looking for segments that cross 'level'. If such a segment is
found, its coordinates are added to a growing list of segments,
which is returned by the function. if vertex_connect_high is
nonzero, high-values pixels are considered to be face+vertex
connected into objects; otherwise low-valued pixels are.

"""
    #if array.shape[0] < 2 or array.shape[1] < 2:
    #    raise ValueError("Input array must be at least 2x2.")

    arc_list = []
    #cdef int n

    # The plan is to iterate a 2x2 square across the input array. This means
    # that the upper-left corner of the square needs to iterate across a
    # sub-array that's one-less-large in each direction (so that the square
    # never steps out of bounds). The square is represented by four pointers:
    # ul, ur, ll, and lr (for 'upper left', etc.). We also maintain the current
    # 2D coordinates for the position of the upper-left pointer. Note that we
    # ensured that the array is of type 'double' and is C-contiguous (last
    # index varies the fastest).

    # Current coords start at 0,0.
    coords = [0,0]
    coords[0] = 0
    coords[1] = 0

    # Calculate the number of iterations we'll need
    if interp == 'crisp':
        num_square_steps = (array.shape[0] - 1) * (array.shape[1] - 1)
    else:
        num_square_steps = (array.shape[1] - 1) * (array.shape[2] - 1)

    square_case = 0
    #top; bottom; left.; right;
    ul=0.; ur=0.; ll=0.; lr=0.
    #r0, r1, c0, c1

    for n in range(num_square_steps):
        # There are sixteen different possible square types, diagramed below.
        # A + indicates that the vertex is above the contour value, and a -
        # indicates that the vertex is below or equal to the contour value.
        # The vertices of each square are:
        # ul ur
        # ll lr
        # and can be treated as a binary value with the bits in that order. Thus
        # each square case can be numbered:
        # 0-- 1+- 2-+ 3++ 4-- 5+- 6-+ 7++
        # -- -- -- -- +- +- +- +-
        #
        # 8-- 9+- 10-+ 11++ 12-- 13+- 14-+ 15++
        # -+ -+ -+ -+ ++ ++ ++ ++
        #
        # The position of the line segment that cuts through (or
        # doesn't, in case 0 and 15) each square is clear, except in
        # cases 6 and 9. In this case, where the segments are placed
        # is determined by vertex_connect_high. If
        # vertex_connect_high is false, then lines like \\ are drawn
        # through square 6, and lines like // are drawn through square
        # 9. Otherwise, the situation is reversed.
        # Finally, recall that we draw the lines so that (moving from tail to
        # head) the lower-valued pixels are on the left of the line. So, for
        # example, case 1 entails a line slanting from the middle of the top of
        # the square to the middle of the left side of the square.

        r0, c0 = coords[0], coords[1]
        r1, c1 = r0 + 1, c0 + 1
        
        if interp == 'crisp':
            ul = array[r0, c0]
            ur = array[r0, c1]
            ll = array[r1, c0]
            lr = array[r1, c1]
        elif interp == 'ensemble' or interp == 'quantile' or interp == 'gmm' or interp == 'g':
            
            print "Cell: " + str(coords)
            
            ul_dist = np.zeros(shape=(MEM))
            ur_dist = np.zeros(shape=(MEM))
            ll_dist = np.zeros(shape=(MEM))
            lr_dist = np.zeros(shape=(MEM))
            
            for idx in range(0,MEM):
                ul_dist[idx] = array[idx][r0][c0]
                ur_dist[idx] = array[idx][r0][c1]
                ll_dist[idx] = array[idx][r1][c0]
                lr_dist[idx] = array[idx][r1][c1]
                
            x = linspace( -50, 50, 600 )
            
            #find peaks...
            worked = [True]*4
            try:
                a = stats.gaussian_kde(ul_dist)
            except:
                print "a kde didn't work @ " + str(coords)
                worked[0] = False
            try:
                b = stats.gaussian_kde(ur_dist)
            except:
                print "b kde didn't work @ " + str(coords)
                worked[1] = False
            try:
                c = stats.gaussian_kde(ll_dist)
            except:
                print "d kde didn't work @ " + str(coords)
                worked[2] = False
            try:
                d = stats.gaussian_kde(lr_dist)
            except:
                print "e kde didn't work @ " + str(coords)
                worked[3] = False
                
            print worked
                
            k = [a , b, c, d]
        
            for idx in range(0,len(k),1):
                _max_t, _min_t = peakdetect(k[idx](x),x,lookahead=2,delta=0)
                
                xm_t = [p[0] for p in _max_t]
                ym_t = [p[1] for p in _max_t]
               
                m = max(ym_t)
                p = ym_t.index(m)
                temp_hp = xm_t[p]
               
                if idx == 0:
                    if worked[0] == True:
                        ul = temp_hp
                    else:
                        print "setting zero"
                        ul = 0.
                elif idx == 1:
                    if worked[1] == True:
                        ur = temp_hp
                    else:
                        print "setting zero"
                        ur = 0.
                elif idx == 2:
                    if worked[2] == True:
                        ll = temp_hp
                    else:
                        print "setting zero"
                        ll = 0.
                else:
                    if worked[3] == True:
                        lr = temp_hp
                    else:
                        print "setting zero"
                        lr = 0.
            

        # now in advance the coords indices
        if interp == 'crisp':
            if coords[1] < array.shape[1] - 2:
                coords[1] += 1
            else:
                coords[0] += 1
                coords[1] = 0
        else:
            if coords[1] < array.shape[2] - 2:
                coords[1] += 1
            else:
                coords[0] += 1
                coords[1] = 0
            


        square_case = 0
        if (ul > level): square_case += 1
        if (ur > level): square_case += 2
        if (ll > level): square_case += 4
        if (lr > level): square_case += 8

        if (square_case != 0 and square_case != 15):
            # only do anything if there's a line passing through the
            # square. Cases 0 and 15 are entirely below/above the contour.

            top = r0, c0 + _get_fraction(ul, ur, level)
            bottom = r1, c0 + _get_fraction(ll, lr, level)
            left = r0 + _get_fraction(ul, ll, level), c0
            right = r0 + _get_fraction(ur, lr, level), c1

            if (square_case == 1):
                # top to left
                arc_list.append(top)
                arc_list.append(left)
            elif (square_case == 2):
                # right to top
                arc_list.append(right)
                arc_list.append(top)
            elif (square_case == 3):
                # right to left
                arc_list.append(right)
                arc_list.append(left)
            elif (square_case == 4):
                # left to bottom
                arc_list.append(left)
                arc_list.append(bottom)
            elif (square_case == 5):
                # top to bottom
                arc_list.append(top)
                arc_list.append(bottom)
            elif (square_case == 6):
                if vertex_connect_high:
                    arc_list.append(left)
                    arc_list.append(top)

                    arc_list.append(right)
                    arc_list.append(bottom)
                else:
                    arc_list.append(right)
                    arc_list.append(top)
                    arc_list.append(left)
                    arc_list.append(bottom)
            elif (square_case == 7):
                # right to bottom
                arc_list.append(right)
                arc_list.append(bottom)
            elif (square_case == 8):
                # bottom to right
                arc_list.append(bottom)
                arc_list.append(right)
            elif (square_case == 9):
                if vertex_connect_high:
                    arc_list.append(top)
                    arc_list.append(right)

                    arc_list.append(bottom)
                    arc_list.append(left)
                else:
                    arc_list.append(top)
                    arc_list.append(left)

                    arc_list.append(bottom)
                    arc_list.append(right)
            elif (square_case == 10):
                # bottom to top
                arc_list.append(bottom)
                arc_list.append(top)
            elif (square_case == 11):
                # bottom to left
                arc_list.append(bottom)
                arc_list.append(left)
            elif (square_case == 12):
                # lef to right
                arc_list.append(left)
                arc_list.append(right)
            elif (square_case == 13):
                # top to right
                arc_list.append(top)
                arc_list.append(right)
            elif (square_case == 14):
                # left to top
                arc_list.append(left)
                arc_list.append(top)
    return arc_list

def find_contours(array, level, fully_connected='low', positive_orientation='low', type='crisp'):
    """Find iso-valued contours in a 2D array for a given level value.

Uses the "marching squares" method to compute a the iso-valued contours of
the input 2D array for a particular level value. Array values are linearly
interpolated to provide better precision for the output contours.

Parameters
----------
array : 2D ndarray of double
Input data in which to find contours.
level : float
Value along which to find contours in the array.
fully_connected : str, {'low', 'high'}
Indicates whether array elements below the given level value are to be
considered fully-connected (and hence elements above the value will
only be face connected), or vice-versa. (See notes below for details.)
positive_orientation : either 'low' or 'high'
Indicates whether the output contours will produce positively-oriented
polygons around islands of low- or high-valued elements. If 'low' then
contours will wind counter- clockwise around elements below the
iso-value. Alternately, this means that low-valued elements are always
on the left of the contour. (See below for details.)

Returns
-------
contours : list of (n,2)-ndarrays
Each contour is an ndarray of shape ``(n, 2)``,
consisting of n ``(x, y)`` coordinates along the contour.

Notes
-----
The marching squares algorithm is a special case of the marching cubes
algorithm [1]_. A simple explanation is available here::

http://www.essi.fr/~lingrand/MarchingCubes/algo.html

There is a single ambiguous case in the marching squares algorithm: when
a given ``2 x 2``-element square has two high-valued and two low-valued
elements, each pair diagonally adjacent. (Where high- and low-valued is
with respect to the contour value sought.) In this case, either the
high-valued elements can be 'connected together' via a thin isthmus that
separates the low-valued elements, or vice-versa. When elements are
connected together across a diagonal, they are considered 'fully
connected' (also known as 'face+vertex-connected' or '8-connected'). Only
high-valued or low-valued elements can be fully-connected, the other set
will be considered as 'face-connected' or '4-connected'. By default,
low-valued elements are considered fully-connected; this can be altered
with the 'fully_connected' parameter.

Output contours are not guaranteed to be closed: contours which intersect
the array edge will be left open. All other contours will be closed. (The
closed-ness of a contours can be tested by checking whether the beginning
point is the same as the end point.)

Contours are oriented. By default, array values lower than the contour
value are to the left of the contour and values greater than the contour
value are to the right. This means that contours will wind
counter-clockwise (i.e. in 'positive orientation') around islands of
low-valued pixels. This behavior can be altered with the
'positive_orientation' parameter.

The order of the contours in the output list is determined by the position
of the smallest ``x,y`` (in lexicographical order) coordinate in the
contour. This is a side-effect of how the input array is traversed, but
can be relied upon.

.. warning::

Array coordinates/values are assumed to refer to the *center* of the
array element. Take a simple example input: ``[0, 1]``. The interpolated
position of 0.5 in this array is midway between the 0-element (at
``x=0``) and the 1-element (at ``x=1``), and thus would fall at
``x=0.5``.

This means that to find reasonable contours, it is best to find contours
midway between the expected "light" and "dark" values. In particular,
given a binarized array, *do not* choose to find contours at the low or
high value of the array. This will often yield degenerate contours,
especially around structures that are a single array element wide. Instead
choose a middle value, as above.

References
----------
.. [1] Lorensen, William and Harvey E. Cline. Marching Cubes: A High
Resolution 3D Surface Construction Algorithm. Computer Graphics
(SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).

"""
    array = np.asarray(array, dtype=np.double)
    #if array.ndim != 2:
    #    raise ValueError('Only 2D arrays are supported.')
    level = float(level)
    if (fully_connected not in _param_options or
       positive_orientation not in _param_options):
        raise ValueError('Parameters "fully_connected" and'
        ' "positive_orientation" must be either "high" or "low".')
    point_list = iterate_and_store(array, level, fully_connected == 'high', interp=type)
    contours = _assemble_contours(_take_2(point_list))
    if positive_orientation == 'high':
        contours = [c[::-1] for c in contours]
    return contours


def _take_2(seq):
    iterator = iter(seq)
    while(True):
        n1 = iterator.next()
        n2 = iterator.next()
        yield (n1, n2)


def _assemble_contours(points_iterator):
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for from_point, to_point in points_iterator:
        # Ignore degenerate segments.
        # This happens when (and only when) one vertex of the square is
        # exactly the contour level, and the rest are above or below.
        # This degnerate vertex will be picked up later by neighboring squares.
        if from_point == to_point:
            continue

        tail_data = starts.get(to_point)
        head_data = ends.get(from_point)

        if tail_data is not None and head_data is not None:
            tail, tail_num = tail_data
            head, head_num = head_data
            # We need to connect these two contours.
            if tail is head:
                # We need to closed a contour.
                # Add the end point, and remove the contour from the
                # 'starts' and 'ends' dicts.
                head.append(to_point)
                del starts[to_point]
                del ends[from_point]
            else: # tail is not head
                # We need to join two distinct contours.
                # We want to keep the first contour segment created, so that
                # the final contours are ordered left->right, top->bottom.
                if tail_num > head_num:
                    # tail was created second. Append tail to head.
                    head.extend(tail)
                    # remove all traces of tail:
                    del starts[to_point]
                    del ends[tail[-1]]
                    del contours[tail_num]
                    # remove the old end of head and add the new end.
                    del ends[from_point]
                    ends[head[-1]] = (head, head_num)
                else: # tail_num <= head_num
                    # head was created second. Prepend head to tail.
                    tail.extendleft(reversed(head))
                    # remove all traces of head:
                    del starts[head[0]]
                    del ends[from_point]
                    del contours[head_num]
                    # remove the old start of tail and add the new start.
                    del starts[to_point]
                    starts[tail[0]] = (tail, tail_num)
        elif tail_data is None and head_data is None:
            # we need to add a new contour
            current_index += 1
            new_num = current_index
            new_contour = deque((from_point, to_point))
            contours[new_num] = new_contour
            starts[from_point] = (new_contour, new_num)
            ends[to_point] = (new_contour, new_num)
        elif tail_data is not None and head_data is None:
            tail, tail_num = tail_data
            # We've found a single contour to which the new segment should be
            # prepended.
            tail.appendleft(from_point)
            del starts[to_point]
            starts[from_point] = (tail, tail_num)
        elif tail_data is None and head_data is not None:
            head, head_num = head_data
            # We've found a single contour to which the new segment should be
            # appended
            head.append(to_point)
            del ends[from_point]
            ends[to_point] = (head, head_num)
    # end iteration over from_ and to_ points

    return [np.array(contour) for (num, contour) in sorted(contours.items())]