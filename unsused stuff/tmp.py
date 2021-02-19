import networkx as nx
from cdlib import algorithms, ensemble, evaluation
from Community_utils import my_grid_search
g = nx.karate_club_graph()
resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)
randomize = ensemble.BoolParameter(name="randomize")
communities, scoring = my_grid_search(graph=g, method=algorithms.louvain,
                                                    parameters=[resolution, randomize],
                                                    quality_score=evaluation.erdos_renyi_modularity,
                                                    aggregate=None)
print(communities, scoring)