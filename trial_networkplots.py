# %%
# 
import networkx as nx
import numpy as np
import string

dt = [('len', float)]

np_data = np.load(r"C:\Users\andre\Documents\EPFL\MA3\Applied Data Analysis\2022\ada-2022-project-enchiladas\states_ba\BEERREL_AVG_euclidean_dm.npy")


print(np_data)
print(np_data.shape)

np_data = np_data.view(dt)


print(np_data)
print(np_data.shape)

# %%


G = nx.from_numpy_matrix(np_data)
G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),string.ascii_uppercase)))    

G = nx.drawing.nx_agraph.to_agraph(G)

G.node_attr.update(color="red", style="filled")
G.edge_attr.update(color="blue", width="2.0")

G.draw(r'C:\Users\andre\Documents\EPFL\MA3\Applied Data Analysis\2022\out.png', format='png', prog='neato')
# %%
