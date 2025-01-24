import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def plot_graph_pomegranate(model):
    edges = [(i, j) for i, parents in enumerate(model.structure) for j in parents]
    G = nx.DiGraph(edges)

    plt.figure(figsize=(10, 8))
    nx.draw_networkx(
        G,
        with_labels=True,
        # labels={i: name for i, name in enumerate(model.states)},
        labels={name for name in model.states},
        node_size=3000,
        node_color="lightblue",
    )
    plt.title("Struktura Sieci Bayesowskiej")
    plt.show()


def generate_data_pomegranate(model, observations):
    prediction = model.predict_proba(observations)

    generated = {}
    for state, prob in zip(model.states, prediction):
        if isinstance(prob, str):
            generated[state.name] = prob
        else:
            generated[state.name] = np.random.choice(
                prob.parameters[0].keys(), p=list(prob.parameters[0].values())
            )
    return generated
