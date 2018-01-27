from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import math
# llcrnrlat,llcrnrlon,urcrnrlat,urcrnrlon
# are the lat/lon values of the lower left and upper right corners
# of the map.
# lat_ts is the latitude of true scale.
# resolution = 'c' means use crude resolution coastlines.
aspect_ratio = (16,9)
fig = plt.figure(figsize=aspect_ratio)
plt.tight_layout()
ax = fig.add_axes([0,0,1,1])
aspect_ratio_inv = aspect_ratio[1]/aspect_ratio[0]
ax.set_aspect(aspect_ratio_inv, adjustable='datalim')

color = {}
color['ocean'] = "#9EDBFF"
color['grass'] = "#C4E8BA"
color['place'] = "#D4A23F"
color['drive'] = "#F44336"
color['flight'] = "#673AB7"
color['train'] = "#FFC107"

nylat = 32.8080055; nylon = -117.2354315
lonlat = 39.9042; lonlon = 116.4074
intlat = (nylat+lonlat)/1.5; intlon = -180
z = 0.5
x = nylon
y = nylat

latl = ((y-z*0.5625))
lath = ((y+z*0.5625))
lonl = ((x-z))
lonh = ((x+z))

lonmin, latmin = (-180, -80)
lonmax, latmax = (180, 80)
lonl = (lonl if lonmax > lonl > lonmin else lonmin)
lonh = (lonh if lonmin < lonh < lonmax else lonmax)
latl = (latl if latmax > latl > latmin else latmin)
lath = (lath if latmin < lath < latmax else latmax)

m = Basemap(projection='merc',llcrnrlat=latl,urcrnrlat=lath,\
            llcrnrlon=lonl,urcrnrlon=lonh,lat_ts=20,resolution='l',
            epsg=4326)
m.arcgisimage(service='World_Street_Map', xpixels = 1920, verbose= True)

# m.drawmapboundary(fill_color=color['ocean'])
# m.fillcontinents(color=color['grass'],lake_color=color['ocean'],alpha=0.5)
xl, yl = m(lonl,latl)
xh, yh = m(lonh,lath)
# m.drawcoastlines()
# m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
# m.drawparallels(np.arange(-90.,91.,30.))
# m.drawmeridians(np.arange(-180.,181.,60.))
# m.drawmapboundary(fill_color='white')
xmin, ymin = m(-180, -80)
xmax, ymax = m(180, 80)
xl = (xl if xmax > xl > xmin else xmin)
xh = (xh if xmin < xh < xmax else xmax)
yl = (yl if ymax > yl > ymin else ymin)
yh = (yh if ymin < yh < ymax else ymax)
ax.set_xlim([xl,xh])
ax.set_ylim([yl,yh])
# draw great circle route between NY and London
m.drawgreatcircle(nylon,nylat,intlon,intlat,linestyle='solid',linewidth=0.5,color=color['flight'])
m.drawgreatcircle(-intlon,intlat,lonlon,lonlat,linestyle='solid',linewidth=0.5,color=color['flight'])
plt.title("Mercator Projection")
plt.show()