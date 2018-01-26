from mpl_toolkits.basemap import Basemap
from math import sin, cos, sqrt, atan2, radians, ceil, floor
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import json
import os
import _pickle as pickle
import hashlib
import warnings

warnings.filterwarnings("ignore")

data_root_dir = 'data'
data_year = '2017'

color = {}
color['ocean'] = "#9EDBFF"
color['grass'] = "#C4E8BA"
color['place'] = "#D4A23F"
color['drive'] = "#F44336"
color['flight'] = "#673AB7"
color['train'] = "#FFC107"

aspect_ratio = (16,9)
render_dpi = 120
render_fps = 60

zoom_init = 5 # Initial zoom width (in degrees longitude)
zoom_length = 200 # Number of animation frames to zoom out for
zoom_post = 500 # Number of animation frames to linger after zooming out

tx = ty = 0.02 # Bottom left coordinate for the date text

M = ['January','February','March','April','May','June','July','August','September','October','November','December']

def BasemapWrapper(*args, **kwargs):
    assert len(args)==0, "Shouldn't have any normal arguments to Basemap..."

    verbose = False
    if "verbose" not in kwargs:
        verbose = False
    else:
        verbose = kwargs["verbose"]
        assert type(verbose) == bool

    cacheDir = None
    if "cacheDir" not in kwargs:
        cacheDir = os.path.join(os.path.expanduser("~"), "Dropbox/data/BasemapUtilsCache/")
    else:
        cacheDir = kwargs["cacheDir"]

    outputBase = os.path.dirname(cacheDir)
    if outputBase!='' and not os.path.exists(outputBase):
        if verbose:
            print("Output directory doesn't exist, making output dirs: %s" % (outputBase))
        os.makedirs(outputBase)

    newKwargs = {}
    kwargsIgnore = ["cacheDir","verbose"]
    for k,v in kwargs.items():
        if k not in kwargsIgnore:
            newKwargs[k] = v


    uniqueRepr = str(set(tuple(newKwargs.items()))).encode('ascii')
    hashedFn = str(hashlib.sha224(uniqueRepr).hexdigest()) + ".p"
    # hashedFn = 'ea02fea0e9025ea82c2d8f8e8babaa37e0855bdb53634a5879a9b645.p'
    # hashedFn = '443935f8398a8154d179cef969d23df70d1ee7283b037149ceed5797.p'
    # hashedFn = '443935f8398a8154d179cef969d23df70d1ee7283b037149ceed5797.p'
    hashedFn = '89694ea432343f426503da7fd75944a6beeb103e190936ebf6678f8e.p'
    newFn = os.path.join(outputBase,hashedFn)

    if os.path.isfile(newFn):
        if verbose:
            print("Loading from file: %s" % (newFn))
        m = pickle.load(open(newFn,'rb'))
    else:
        if verbose:
            print("Creating object and saving to file: %s" % (newFn))
        m = Basemap(*args, **newKwargs)
        pickle.dump(m,open(newFn,'wb'),-1)
    return m

## Functions
def distance(start, end):
    lon1 = radians(start[0])
    lat1 = radians(start[1])
    lon2 = radians(end[0])
    lat2 = radians(end[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 3960 * c

sign = lambda x: (1, -1)[x < 0]

def roundup(x):
    return int(ceil(x / 1000)) * 1000

def datafiles(root, year):
   for dirpath,_,filenames in os.walk(os.path.abspath(os.path.join(root, year))):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

## Initialize the figure
fig = plt.figure(figsize=aspect_ratio)
plt.tight_layout()
ax = fig.add_axes([0,0,1,1])

## Set up Basemap
basemap = BasemapWrapper(
        verbose=True,
        projection='merc',
        llcrnrlon=-180,
        llcrnrlat=-80,
        urcrnrlon=180,
        urcrnrlat=80,
        lat_ts=20,
        fix_aspect=False,
        resolution='l')
basemap.drawmapboundary(fill_color=color['ocean'])
basemap.fillcontinents(color=color['grass'],lake_color=color['ocean'])
basemap.etopo(alpha=0.25)
map_states = [basemap.drawstates()]
map_countries = [basemap.drawcountries()]
bl_text = ax.text(tx, ty, '',
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
            color='black',
            fontsize='xx-large')

## Configure aspect ratio
aspect_ratio_inv = aspect_ratio[1]/aspect_ratio[0]
ax.set_aspect(aspect_ratio_inv, adjustable='datalim')

## Import city data
with open('places.json', 'r') as file_places:
    data_places = json.load(file_places)

## Import data and sort by timestamp
data = []
for f in datafiles(data_root_dir, data_year):
    with open(f, 'r') as i:
        v = json.load(i)['locations']
        for k, d in enumerate(v):
            d['time'] = int(d['timestampMs']) // 1000
            d['lat'] = d['latitudeE7'] / 10000000
            d['lon'] = d['longitudeE7'] / 10000000
            d['is_inter'] = False
            date =  datetime.datetime.fromtimestamp(d['time'])
            d['date'] = '{} {}, {}'.format(M[date.month-1], date.day, date.year)
            d['place'] = None
            del d['timestampMs']
            del d['latitudeE7']
            del d['longitudeE7']
        data += v
data = sorted(data, key=lambda k: k['time'])

data_min = 1

## Compute histogram of places visited
hist_places = {}
for data_point in data:
    coor = (data_point['lon'],data_point['lat'])
    nearest_place = min(data_places, key=lambda x: distance((x['lon'],x['lat']), coor))
    if distance((nearest_place['lon'],nearest_place['lat']), coor) < 20:
        data_point['place'] = nearest_place['name']
    if data_point['place'] in hist_places:
        hist_places[data_point['place']] += 1
    else:
        hist_places[data_point['place']] = 1

## Scrub places with over 200 consecutive data points in one city
last_place_visited = None
flag_scrub = True
if flag_scrub:
    print("# data points before scrubbing =",len(data))
    visit_count = 0
    hist_scrub = {}
    p = 0
    while p < len(data):
        if not data[p]['is_inter']:
            if data[p]['place'] != None:
                if last_place_visited != data[p]['place']:
                    visit_count = 1
                    last_place_visited = data[p]['place']
                else:
                    visit_count += 1
                if visit_count > 200:
                    if data[p]['place'] in hist_scrub:
                        hist_scrub[data[p]['place']] += 1
                    else:
                        hist_scrub[data[p]['place']] = 1
                    del data[p]
        p += 1
    print("# data points after scrubbing =",len(data))
    print("Places scrubed:",[(hist_scrub[k],k) for k in sorted(hist_scrub, key=hist_scrub.get, reverse=True)])

## Loop through data to calculate points of transit (flights, trains, etc.)
flag_animate_transit = True
if flag_animate_transit:
    i = data_min
    interpolation_list = [] # 0 is the data index, 1 is the type, 2 is the zipped (x,y) tuples
    while i < len(data):
        if not data[i]['is_inter']:
            x, y = (data[i]['lon'], data[i]['lat'])
            xp, yp = (data[i-1]['lon'], data[i-1]['lat'])

            if distance((x,y),(xp,yp)) > 500: # (500 miles) Flight
                if (sign(x) != sign(xp)) and (abs(x)+abs(xp)>180):
                    xi, yi = (sign(x)*180, (y+yp)/1.25)
                    line, = basemap.drawgreatcircle(x,y,xi,yi)
                    fx, fy = line.get_data(); line.remove(); del line; fc1 = list(zip(fx[:-1],fy[:-1]));
                    line, = basemap.drawgreatcircle(-xi,yi,xp,yp)
                    fx, fy = line.get_data(); line.remove(); del line; fc2 = list(zip(fx[:-1],fy[:-1]));
                    interpolation_list.append({
                        'index': i-1,
                        'length': len(fc1),
                        'date': data[i-1]['date'],
                        'type': "flight",
                        'coordinates': fc1+fc2}); del fc1; del fc2;
                else:
                    line, = basemap.drawgreatcircle(x,y,xp,yp)
                    fx, fy = line.get_data(); line.remove(); del line; fc = list(zip(fx[:-1],fy[:-1]));
                    interpolation_list.append({
                        'index': i-1,
                        'length': len(fc),
                        'date': data[i-1]['date'],
                        'type': "flight",
                        'coordinates': fc}); del fc;
            elif distance((x,y),(xp,yp)) > 50: # (50 miles) Train
                if x > 0:
                    line, = basemap.drawgreatcircle(x,y,xp,yp)
                    fx, fy = line.get_data(); line.remove(); del line; fc = list(zip(fx[:-1],fy[:-1]));
                    interpolation_list.append({
                        'index': i-1,
                        'length': len(fc),
                        'date': data[i-1]['date'],
                        'type': "train",
                        'coordinates': fc}); del fc;
        i += 1

    print("# data points before transit interpolation =",len(data))
    i = 0
    while i < len(interpolation_list):
        index = interpolation_list[i]['index']
        if interpolation_list[i]['type'] is "flight":
            print("Interpolating flight (at index {} using {} data points)...".format(index, interpolation_list[i]['length']))
        if interpolation_list[i]['type'] is "train":
            print("Interpolating train ride (at index {} using {} data points)...".format(index, interpolation_list[i]['length']))
        for coordinate in interpolation_list[i]['coordinates']:
            point = {}; point['is_inter'] = True;
            point['inter_index'] = i;
            point['inter_frame'] = interpolation_list[i]['index'];
            point['inter_length'] = interpolation_list[i]['length'];
            point['inter_type'] = interpolation_list[i]['type'];
            point['x'] = coordinate[0]; point['y'] = coordinate[1]; point['date'] = data[index]['date'];
            point['lon'] = data[index+1]['lon']; point['lat'] = data[index+1]['lat'];
            data.insert(index+1, point)
        for j in range(i+1,len(interpolation_list)):
            ## Shift all upcoming interpolation indexes by the length of this interpolation
            interpolation_list[j]['index'] += interpolation_list[i]['length']
        i += 1
    print("# data points after transit interpolation =",len(data))

## Pad the data with duplicates for before & after zoom effects
zoom_start_frame = len(data)
data += [data[-1]]*zoom_post
data = [data[0]]*(zoom_post//8+1) + data

## Configure zoom
xmin, ymin = basemap(-180, -80)
xmax, ymax = basemap(180, 80)
xmin = ceil(xmin); ymin = ceil(ymin); xmax = floor(xmax); ymax = floor(ymax);
z = zoom_init
x, y = (data[0]['lon'], data[0]['lat'])
xl, yl = basemap(x-z, y-z*aspect_ratio_inv)
xh, yh = basemap(x+z, y+z*aspect_ratio_inv)
xw = xh - xl
yw = yh - yl

## Create a legend
legend=[]
legend.append(mpatches.Patch(color=color['drive'], label='By Car / Foot'))
legend.append(mpatches.Patch(color=color['train'], label='By Train'))
legend.append(mpatches.Patch(color=color['flight'], label='By Plane'))
plt.legend(handles=legend, loc="upper left", fontsize='xx-large')

## Map cities
map_places = []
for i, c in enumerate(data_places):
    cx, cy = basemap(c['lon'],c['lat'])
    map_places.append(ax.plot(cx,cy,'go',
        color=color['place'],
        marker='o',snap=True,
        markersize=6)[0])
    map_places.append(ax.text(s=c['name'],
        x=cx-10000,y=cy+5000,
        horizontalalignment='right',
        multialignment='center',
        color="black",alpha=1,fontsize=12))

trail = []

with open(os.path.join(data_root_dir,data_year+'.json'),'w') as fp:
    json.dump(data, fp, indent=4)

sys.exit(-1)

## Define animation function
def update(i):

    global z,xl,xh,yl,yh,xw,yw

    ## Draw trail
    if data[i]['is_inter']: # if interpolated (greatcircle)
        x, y = (data[i]['x'], data[i]['y'])
        if data[i-1]['is_inter']:
            xp, yp = (data[i-1]['x'], data[i-1]['y'])
        else:
            xp, yp = basemap(data[i-1]['lon'], data[i-1]['lat'])
        if (abs(x-xp) > xw) or (abs(y-yp) > yw):
            xp = x; yp = y;
        trail.append(basemap.plot([x,xp], [y,yp], color=color[data[i]['inter_type']], linewidth=2, latlon=False)[0])
    elif not data[i]['is_inter']: # if not interpolated
        x, y = basemap(data[i]['lon'], data[i]['lat'])
        if not data[i-1]['is_inter']:
            xp, yp = basemap(data[i-1]['lon'], data[i-1]['lat'])
        else:
            xp, yp = (data[i-1]['x'], data[i-1]['y'])
        trail.append(basemap.plot([x,xp], [y,yp], color=color['drive'], linewidth=2, latlon=False)[0])

    ## Adjust zoom
    if (zoom_start_frame < i < zoom_start_frame + zoom_length):
        d = i - zoom_start_frame
        z = d*1.5 + zoom_init
        a = 1 - d/(zoom_length/10)
        a = a if a > 0 else 0
        [t.set_alpha(a) for t in map_places]
        [t.set_alpha(a) for t in map_states]
        [t.set_alpha(a) for t in map_countries]
        [t.set_linewidth(2 - 1.75*d/zoom_length) for t in trail]

    ## Adjust axis mins and maxs
    if (i < zoom_start_frame + zoom_length):
        if not data[i]['is_inter']:
            x, y = (data[i]['lon'], data[i]['lat'])
            xl, yl = basemap(x-z, y-z*aspect_ratio_inv)
            xh, yh = basemap(x+z, y+z*aspect_ratio_inv)
        else:
            xl, yl = (x-xw/2,y-yw/2)
            xh, yh = (x+xw/2,y+yw/2)
        xl = (xl if xmax > xl > xmin else xmin)
        xh = (xh if xmin < xh < xmax else xmax)
        yl = (yl if ymax > yl > ymin else ymin)
        yh = (yh if ymin < yh < ymax else ymax)
    else:
        xl = xmin; xh = xmax; yl = ymin; yh = ymax;

    ## Bound zoomed view by Mercator bounds
    ax.set_xlim([xl,xh])
    ax.set_ylim([yl,yh])
    # print(
    #     (xl if xmax > xl > xmin else "xmin"),
    #     (xh if xmin < xh < xmax else "xmax"),
    #     (yl if ymax > yl > ymin else "ymin"),
    #     (yh if ymin < yh < ymax else "ymax"))

    ## Update text in the bottom-left corner
    bl_date = data[i]['date']
    if (data[i]['is_inter']):
        if (data[i]['inter_type'] == 'flight'):
            bl_place = "Flying..."
        if (data[i]['inter_type'] == 'train'):
            bl_place = "Riding the train..."
    else:
        if (data[i]['place'] is not None):
            bl_place = data[i]['place']
        else:
            bl_place = "Driving..."

    if (i < zoom_start_frame ):
        bl_text.set_text("{} — {} — {} — {} — {}".format(bl_date, bl_place, i, data[i]['is_inter'], data[i-1]['is_inter']))
    else:
        bl_text.set_text("{}".format(bl_date))

    print('{} frame{plural} remaining...'.format(max(frames)-i,plural=("" if (max(frames)-i)==1 else "s")))

    ## Return animated handlers
    return bl_text, ax

## Set animation frames
frames = np.arange(len(data)-3060,len(data)-2700)
assert frames[0] > 0
assert zoom_start_frame < len(data)
print("# of animation frames:",len(frames))

## Create and save animation
anim = animation.FuncAnimation(fig, update, frames=frames)
writer = animation.writers['ffmpeg'](fps=render_fps)
anim.save('map_'+str(int(time.time()))+'.mp4', dpi=render_dpi, writer=writer)

## Report stats
print("Top 3 Places Visited:")
top_cities = [(hist_places[k],k) for k in sorted(hist_places, key=hist_places.get, reverse=True) if k is not None]
print("\t1st >>",top_cities[0][1],"\n","\t2nd >>",top_cities[1][1],"\n","\t3rd >>",top_cities[2][1])