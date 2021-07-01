import osmnx as ox
import rasterio.plot
from matplotlib import pyplot as plt, image as mimage

from .tile_coordinates_helper import num2deg, get_extent

WIDE_ROADS = 14
MID_ROADS = 12
SMALL_ROAD = 8
# https://wiki.openstreetmap.org/wiki/Key:highway
road_width_dict = {
    "motorway": WIDE_ROADS,
    "trunk": WIDE_ROADS,
    "primary": WIDE_ROADS,
    "secondary": WIDE_ROADS,
    "tertiary": WIDE_ROADS,
    "unclassified": MID_ROADS,
    "residential": MID_ROADS,

    # Link roads
    "motorway_link": MID_ROADS,
    "trunk_link": MID_ROADS,
    "primary_link": MID_ROADS,
    "secondary_link": SMALL_ROAD,
    "tertiary_link": SMALL_ROAD,

    # special road types
    "living_street": SMALL_ROAD,
    "service": SMALL_ROAD,  # parking, ...
    "pedestrian": SMALL_ROAD,
    "track": 0,
    "bus_guideway": 0,
    "escape": 0,
    "raceway": MID_ROADS,
    "road": SMALL_ROAD,
    "busway": SMALL_ROAD,

    # paths
    "footway": 0,
    "bridleway": 0,
    "steps": 0,
    "corridor": 0,
    "path": 0,
}


def apply_street_width(highway):
    # all street widths not specified are set to the default value
    default_street_width = 26

    if isinstance(highway, list):
        widths = [road_width_dict.get(road_type, default_street_width) for road_type in highway]
        return max(widths)
    else:

        return road_width_dict.get(highway, default_street_width)


def get_line_widths(way_type):
    # determine line widths
    line_widths = way_type.copy()
    line_widths = line_widths.reset_index(drop=True)

    line_widths = line_widths.apply(apply_street_width)  # lambda x: street_width_dict.get(x, default_street_width)
    return line_widths


def get_osm_roads(x, y, zoom, nr_tiles_width, nr_tiles_height, file):
    # get also surrounding tiles such that the road edges on the boarder of the image are not cut short
    NR_TILES_EXTENDED = 2
    # get top left corner of top left tile
    lat, lng = num2deg(x - NR_TILES_EXTENDED, y - NR_TILES_EXTENDED, zoom)
    # get bottom right corner of bottom right tile
    lat_, lng_ = num2deg(x + nr_tiles_width + NR_TILES_EXTENDED, y + nr_tiles_height + NR_TILES_EXTENDED, zoom)

    G = ox.graph_from_bbox(lat, lat_, lng, lng_,
                           simplify=True,  # if True masks are smoother
                           clean_periphery=False,
                           truncate_by_edge=False,
                           network_type='drive_service')  # all_private, all, drive_service, drive

    graph_gdfs = ox.graph_to_gdfs(G, nodes=False, node_geometry=True, fill_edge_geometry=True)

    DPI = 100

    # magic number: a tile has size 256x256
    height = 256 * nr_tiles_height
    width = 256 * nr_tiles_width

    fig, ax = plt.subplots(figsize=(height / DPI, width / DPI), dpi=DPI)

    # if it is a tiff image we could read the coordinates (lng, lat) from that file with rasterio
    use_rasterio = False
    if use_rasterio:
        # read extent from tiff image
        raster = rasterio.open("test.tiff")
        DEBUG = False
        if DEBUG:
            ax = rasterio.plot.show(raster, ax=ax)
        extent = rasterio.plot.plotting_extent(raster)
    else:
        extent = get_extent(x, y, zoom, nr_tiles_width, nr_tiles_height)

    # get graph edges
    gdf_edges = graph_gdfs["geometry"]

    # get line widths
    line_widths = get_line_widths(graph_gdfs["highway"])

    # plot edges
    ax = gdf_edges.plot(ax=ax, facecolor='none', edgecolor='white', linewidths=line_widths)

    # fix the view to the image
    #  create a dummy image to easily set extent
    im = mimage.AxesImage(ax)
    im.set_extent(extent=extent)
    im.set_visible(False)
    ax.add_image(im)

    # turns off axes
    plt.axis("off")
    # gets rid of the white border
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    # save fig with black background
    plt.savefig(file, pad_inches=0, dpi=DPI, facecolor='black')  # , transparent=True
    plt.close(fig)
