# Copyright 2018-2020 Institute of Neuroscience and Medicine (INM-1), Forschungszentrum Jülich GmbH

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sys import stdout, stdin, platform
from os import popen

if stdin.isatty() and not platform.startswith('win'):
    # running interactively
    _,TERMWIDTH = [int(v) for v in popen('stty size', 'r').read().split()]
else:
    TERMWIDTH=80

# Ascii formatting, see 
# https://stackoverflow.com/questions/8924173/how-do-i-print-bold-text-in-python
class MarkerStyles:
    THIN = u'\u2584'
    MEDIUM = u'\u2586'
    THICK = u'\u2587'
    FULL = u'\u2588'

class ColorStyles:
   WHITE = '\033[37m'
   LIGHTGRAY = '\033[97m'
   GRAY = '\033[2m'
   DARKGRAY = '\033[90m'
   ORANGE = '\033[33m'
   LIGHTBLUE = '\033[94m'
   BLUE = '\033[34m'
   GREEN = '\033[32m'
   RED = '\033[91m'
   DARKRED = '\033[1m'
   END = '\033[0m'

class FontStyles:
   BOLD = '\033[1m'
   BLACK = '\033[90m'
   RED = '\033[91m'
   UNDERLINE = '\033[4m'
   ITALIC = '\033[3m'
   END = '\033[0m'

class Cursor:
    COL0 = '\033[1000D'

def bar(length,
        colorstyle=ColorStyles.DARKGRAY,
        marker=MarkerStyles.THICK,
        shift=0):
    """ 
    Prints a vertical bar with the given color and length to the console.
    Choose markers from https://en.wikipedia.org/wiki/Block_Elements
    """
    #start = Cursor.COL0 if startleft else ""
    start = u"\033[{n}C".format(n=shift-1)
    end = ""#"\n" if close_line else ""
    return start+colorstyle+marker*length+ColorStyles.END+end
    #stdout.flush()

def calibrate(values,labels,maxbars=TERMWIDTH):
    """ calculate scale and shift for bar plot groups """
    minsum = min(float(min(values)),0)
    maxsum = max(float(max(values)),0)
    labelwidth = max([len(str(l)) for l in labels])
    barwidth = maxbars-labelwidth-14
    s = (maxsum-minsum) / float(barwidth) 
    if s==0:
        blocks = lambda x: 0
    else:
        blocks = lambda x : int( (x-minsum) / s + .5)
    negative = -min(minsum,0)
    return (blocks,barwidth,negative,labelwidth)

def format_row(label,value,calibration):
    blocks,maxbars,negative,labelwidth = calibration
    result = "{0:{w}s} {1:10.2f}".format(label,value,w=labelwidth)
    if value<0:
        result += bar(blocks(value),ColorStyles.LIGHTGRAY)
        result += bar(blocks(0)-blocks(value),ColorStyles.GRAY)
        result += bar(maxbars-blocks(0),ColorStyles.LIGHTGRAY,shift=1)
    else:
        result += bar(blocks(0),ColorStyles.LIGHTGRAY)
        result += bar(blocks(value-negative),shift=1)
        result += bar(maxbars-blocks(0)-blocks(value-negative),ColorStyles.LIGHTGRAY)
    return result+" "

