from threading import Thread


class KernelExecutor(Thread):

    def __init__(self, kernel, K, graphs, i):
        Thread.__init__(self)
        self.kernel_similarity = kernel
        self.k_matrix = K
        self.am_graphs = graphs
        self.idx = i
        self.v = len(self.am_graphs)

    def run(self):
        for j in range(self.v):
            self.k_matrix[self.idx, j] = self.kernel_similarity(self.am_graphs[self.idx], self.am_graphs[j])


class FWExecutor(Thread):

    def __init__(self, graph_, shortest_path_method_, out_, idx_):
        Thread.__init__(self)
        self.graph = graph_
        self.shortest_path_method = shortest_path_method_
        self.out = out_
        self.idx = idx_

    def run(self):
        self.out[self.idx] = self.shortest_path_method(self.graph)
