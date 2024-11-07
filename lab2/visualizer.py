import solution_utils
from gmplot import gmplot
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


class Visualizer:
    def __init__(self, data, solution):
        self.data = data
        self.solution = solution
        self.latitudes = []
        self.longitudes = []

    def _extract_coordinates(self):
        geolocator = Nominatim(user_agent="tsp_visualizer")
        for city in solution_utils.decode_solution(self.data, self.solution):
            try:
                location = geolocator.geocode(city)
                if location:
                    self.latitudes.append(location.latitude)
                    self.longitudes.append(location.longitude)
                else:
                    raise ValueError(f"Could not geocode city: {city}")
            except GeocoderTimedOut:
                raise ValueError(f"Geocoding timed out for city: {city}")

    def draw_route_on_map(self):
        self._extract_coordinates()
        gmap = gmplot.GoogleMapPlotter(self.latitudes[0], self.longitudes[0], 10)
        gmap.plot(self.latitudes, self.longitudes, "cornflowerblue", edge_width=2)
        gmap.draw("tsp_route.html")
