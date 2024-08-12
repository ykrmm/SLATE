class Discrete_graph:
    """
    Class representing a temporal graph at a time t
    """
    def __init__(self,dynamic_graph,weights,time,x,score_mat) -> None:
        self.edge_index = dynamic_graph
        self.edge_weight = weights
        self.time = time
        self.x = x
        self.score_mat = score_mat

