import numpy as np


class Weisfeiler_Lehman_Kernel():
    n_labels = 0
    compressed_labels = {} 

    def get_nodes_degree(self, graph):
        v = graph.shape[0]
        ones = np.ones((v, 1))
        return np.dot(graph, ones)

    def get_graphs_labels(self, graphs):
        n = len(graphs)
        graphs_labels = []
        for G in graphs:
            graphs_labels.append(self.get_nodes_degree(G))
        return graphs_labels

    def labels_to_feature_vectors(self, graphs_degree_labels):
        n = len(graphs_degree_labels)
        size = int(np.max(np.concatenate(graphs_degree_labels)))
        degree_component = np.zeros((n, size))
        for i in range(len(graphs_degree_labels)):
            for j in graphs_degree_labels[i]:
                degree_component[i, int(j) - 1] += 1
        return degree_component

    def get_multiset_label(self, graph, graph_labels):
        n = graph.shape[0]
        graphs_labels = np.empty((n,), dtype=np.object)

        for v in range(n):
            np.insert(np.nonzero(graph[v]), 0, values=v)
            neighbors = np.insert(np.nonzero(graph[v]), 0, v)
            multiset = [graph_labels[neighbor][0] for neighbor in neighbors]
            multiset[1:] = np.sort(multiset[1:])
            graphs_labels[v] = np.array(multiset)
        return graphs_labels

    def get_multisets_labels(self, graphs, graphs_labels):
        n = len(graphs)
        multi_labels = np.empty((n,), dtype=np.object)
        for idx in range(n):
            multi_labels[idx] = self.get_multiset_label(graphs[idx], graphs_labels[idx])
        return multi_labels

    def labels_compression(self, multisets_graph_labels):
        graph_cmpr_labels = {}  # {hash_key: [hash, #occurences]}

        for m_labels in multisets_graph_labels:
            str_label = str(m_labels)
            if str_label not in self.compressed_labels:
                self.compressed_labels.update({str_label: self.n_labels})
                self.n_labels += 1

            label_hash = self.compressed_labels[str_label]

            if str_label not in graph_cmpr_labels:
                graph_cmpr_labels.update({str_label: [label_hash, 1]})
            else:
                value = graph_cmpr_labels[str_label]
                value[1] += 1;
                graph_cmpr_labels.update({str_label: value})
        return graph_cmpr_labels

    def relabelling_graphs(self, new_labels, graphs_labels):
        n_graphs = len(new_labels)
        for i in range(n_graphs):
            cmpr_labels = self.labels_compression(new_labels[i])
            n = new_labels[i].shape[0]

            for v in range(n):
                node_labels = new_labels[i][v]
                f_node_labels = cmpr_labels[str(node_labels)][0]
                graphs_labels[i][v] = f_node_labels

    def wl_test_graph_isomorphism(self, graphs, h):
        n = len(graphs)
        # features = np.empty((n,1), dtype = np.object)
        graphs_labels = self.get_graphs_labels(graphs)
        degree_component = self.labels_to_feature_vectors(graphs_labels)
        for i in range(h):
            self.compressed_labels = {}
            self.n_labels = 0
            new_labels = self.get_multisets_labels(graphs, graphs_labels)
            self.relabelling_graphs(new_labels, graphs_labels)
            # print("h: ",str(i))
        return np.array(graphs_labels), degree_component

    def extract_features_vectors(self, wl):
        n = wl.shape[0]
        features = np.zeros((n, self.n_labels))
        for i in range(n):
            for label in wl[i]:
                features[i, int(label)] += 1
        return features

    def normalize(self, X):
        norms = np.sqrt((X ** 2).sum(axis=1, keepdims=True))
        XX = X / norms
        return XX

    def eval_similarities(self, graphs, h):
        self.compressed_labels = {}
        self.n_labels = 0
        WL, degree_component = self.wl_test_graph_isomorphism(graphs, h)
        X = self.extract_features_vectors(WL)
        X = np.concatenate((degree_component, X), axis=1)
        XX = self.normalize(X)
        return np.dot(XX, XX.T)
