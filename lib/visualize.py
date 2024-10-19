# 모델 Visualize

from sklearn.tree import export_graphviz
from lib.get_data import feature_cols
import pydotplus

def visualize(model):
    dot_data = export_graphviz(model, out_file=None, 
                            feature_names=feature_cols,  
                            class_names=['solar', 'wind'],
                            filled=True, rounded=True,  
                            special_characters=True)  
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('models/visualized.png')