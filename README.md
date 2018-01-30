# Mappr

A Python script to render a video of a marker moving on a map showing where you've been, using [your Google location history](https://maps.google.com/locationhistory "Google Timeline").

## Getting Started

### Install Python 3

Be sure to download the [latest version of Python 3](https://www.python.org/downloads/). After the installation, verify that both "python" and "pip" are available to you in the command line. You can do this by entering "which pip" to see if the Terminal returns a valid system path.

### Clone Repository

Enter this command in the Terminal at the location of your choice to copy the Mappr project files there.

```bash
git clone https://github.com/actuallyaswin/mappr.git
```

### Install Prerequisites

Once inside of the Mappr directory, run the following command in Terminal.

```bash
pip install -r requirements.txt
```

All Python dependencies should install except for Basemap.

#### Installing Basemap

1. Download the latest source code zip from the [Basemap Github page](https://github.com/matplotlib/basemap/releases/). Un-zip the archive and navigate your Terminal to the "geos-x.x.x" folder inside of the "basemap-x.x.x" folder.

2. You need to install the GEOS library by compiling it in C and writing the header files to a reasonable location.

```bash
export GEOS_DIR=<where you want the libs and headers to go>
# A reasonable choice on a Unix-like system is /usr/local, or
# if you don't have permission to write there, your home directory.
./configure --prefix=$GEOS_DIR
make; make install
```

3. Go back up to the "basemap-x.x.x" folder and run the following.

```bash
python setup.py install
```

4. Verify that Basemap successfully installed by importing the module. If Basemap successfully imports, the Terminal should return nothing.

```bash
python -c "from mpl_toolkits.basemap import Basemap"
```

### Dataset

This project specifically works using JSON data provided by Google. Location History is a service provided by Google for smartphone users to periodically log their GPS. You can [read more here](https://support.google.com/accounts/answer/3118687?hl=en) on how you can enable/disable the service and begin collecting data. To check if you have data available for use, visit [your Google Timeline](https://maps.google.com/locationhistory "Google Timeline").

1. Once you have verified that Google has GPS data from your device, visit [the Google Takeout page](https://takeout.google.com/settings/takeout) to download your Location History. Be sure to uncheck all other services, and to set the Location History output to JSON. Then click the NEXT button, set the file type to .zip, and click CREATE ARCHIVE.

<div style="text-align:center">
	<img src="https://i.imgur.com/W9Cabtm.png" alt="Screenshot of Google Takeout" width="480px"><br />
	<small><em>Fig 1. Screenshot of Google Takeout</em></small>
</div>

2. Download the archived data (which should have been emailed to you). Unzip the archive, and navigate to the Location History folder, which should contain a single file within titled _Location History.json_.

3. At this point, I recommend splitting the data file into various chunks so that the data is easier to handle, process in Python, and edit manually in text files. The split JSON files should then be moved to the Mappr directory, inside of the `data` folder.

4. Inside of the Location History files, the GPS data is stored within JSON objects, with the first key for the timestamp in milliseconds. You can use [online epoch converter tools](https://www.epochconverter.com/) to determine where certain date ranges start and end. For example, Year 2018 began at timestampMs=1514764800000 and Year 2017 began at timestampMs=1483228800000. It is recommended that you split the Location History JSON into chunked files within folders named by year, all within the `data` directory (such as `mappr/data/2017`, `mappr/data/2018`, etc.).

## Configuration

In the root Mappr folder, you can make changes to `config.ini` to change parameters necessary to create the video.

### Data
| parameter | type    | description                                      |
| --------- | ------- | ------------------------------------------------ |
| `root`    | Folder  | the root folder for all of your location and place data |
| `year`    | Folder  | the specific folder within the root location to use for rendering the video |
### Processing
| parameter | type    | description                                      |
| --------- | ------- | ------------------------------------------------ |
| `debug`    | Boolean  | to print verbose information about Mappr (recommended: yes) |
| `scrub`    | Boolean  | to scrub consecutive data points in one place (dependent on FPS) (recommended: yes) |
| `interpolate`    | Boolean  | to interpolate (animate) flight/train path between points (recommended: yes) |
### Map
| parameter | type    | description                                      |
| --------- | ------- | ------------------------------------------------ |
| `zoom`    | Float  | The width (in degrees longitude / meridians) of the video viewport |
| `use_arcgis`    | Boolean  | to use ARCGIS map background (recommended: no) |
| `use_etopo`    | Boolean  | to use topographic map background (recommended: yes) |
| `use_fill`    | Boolean  | to use solid color-filled map background (recommended: yes) |
### Render
| parameter | type    | description                                      |
| --------- | ------- | ------------------------------------------------ |
|`fps` | Integer | the number of frames (data points) per second in the output video |
|`resolution_w` | Integer | video width (in pixels) |
|`resolution_h` | Integer | video height (in pixels) |
|`time_before` | Integer | time to linger before beginning the animation (in seconds) |
|`time_after` | Integer | time to linger once the animation completes (in seconds) |
|`time_zoom` | Integer | time to zoom out once the animation completes (in seconds) |
|`show_date` | Boolean | to show the calendar date on the bottom left |
|`show_status` | Boolean | to show the location/status on the bottom left |
### Colors
| parameter | type    | description                                      |
| --------- | ------- | ------------------------------------------------ |
|`water` | Hex Color | used for oceans and lakes |
|`grass` | Hex Color | used for land |
|`place` | Hex Color | used for city/park/place markers |
|`drive` | Hex Color | used for the primary map trail (by driving or on foot) |
|`train` | Hex Color | used for map trails generated by train rides |
|`flight` | Hex Color | used for map trails generated by flights |

## Running Mappr

To run Mappr, simply enter the following into the Terminal.

```bash
python mappr.py
```

The resulting video will be placed in the same location uniquely named using the current timestamp. Thus, running the script multiple times will _not_ overwrite previous renderings.

## Suggestions

* Rendering 11871 data frames at 45 FPS with a resolution of 1920x1080 pixels takes about 6 hours on my laptop. I _strongly_ suggest rendering at a much lower resolution (say, 640x360 pixels) first to see how the video looks, then to render the video once more with the maximum settings. If you have a GPU at your disposal, more power to you.

* Use the following formula to estimate the duration of the video:

```text
Time Before + (Total_Frames / FPS) + Time Zoom + Time After
```

## Sample Output

<div style="text-align:center">
	<img src="https://i.imgur.com/HWmnmCN.png" alt="Screenshot of Terminal Output" width="480px"><br />
	<small><em>Fig 2. Screenshot of Terminal Output</em></small>
</div>

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* My parents for allowing me to globetrot
* Nathan Handler for inspiring the project