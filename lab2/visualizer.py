import solution_utils
import pandas as pd
import matplotlib.pyplot as plt
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

    def generate_table(self, output_image, best_solutions, min_values,max_values, std_devs, means, pop_size):
        rows = list(range(100, pop_size+1, 100))
       
        data = {
            'best_solution': [best_solution.evaluation for best_solution in best_solutions],
            'min': min_values,
            'max': max_values,
            'std_dev': std_devs,
            'mean': means
        }

        df = pd.DataFrame(data, index=rows)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

        plt.savefig(output_image, bbox_inches='tight')
        plt.close()
