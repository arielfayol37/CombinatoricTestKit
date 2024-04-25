import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
from gates import *

class Test:
    def __init__(self, name=None):
        self.name = name

    def test_function(self):
        """
        
        ------THIS IS THE WHERE THE USER IS EXPECTED TO INPUT THEIR CIRCUIT
        
        """
        NUMBER_OF_INPUTS = 5 # change this line to modify the number of inputs
        
        x = self.setup(NUMBER_OF_INPUTS)

        # remember that python is 0-indexed.
        # so input 1 is x[0], input 2 is x[1], ..., input n is x[n-1]

        # you can either build the gates one step at a time, do everything in one line, or anywhere in between.

        # Doing one step at a time (preferable, since it will be easier to debug):
        
        and_a = And([x[0], x[1]]) # NOTE: the variable names do not matter. 
        and_b = And([x[2], x[3]]) # if you want to specifiy a name for your gate, then pass it as an argument
        not_b = Not([and_b])      # For example, the variabl z is a gate, with then name 'Z'
        or_a = Or([not_b, and_a]) # NOTE: even though the NOT gate only has one input, it should still be passed using a list.
        or_b = Or([and_b, x[4]])
        z = And([or_a, or_b], name='Z')

        # NOTE: You can do everything in one line (shorter but harder to debug):
        # z = And([Or([Not([And([x[2], x[3]])]), And([x[0], x[1]])]), Or([And([x[2], x[3]]), x[4]])])

        """
        ------- END OF USER INPUT
        """
        

        self.get_result(z)

    def draw_graph(self, directed_graph):
        G = directed_graph
        
        # Create a larger figure with adjusted subplot parameters to take up more space
        plt.figure(figsize=(14, 8))  # Adjust the figure size for better visibility
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust subplot bounds for 70% occupation
        
        # Create the subplot
        # subax1 = plt.subplot(121)
        options = {
            'node_color': 'red',
            'node_size': 500,
            'with_labels': True,
            'width': 3,
        }
        
        # Draw the graph with labels
        nx.draw_spring(G, **options)
        
        # Show the plot
        plt.show()

    def get_result(self, output_node):
        DG = nx.DiGraph()
        GDG = graphviz.Digraph('Circuit Graph', filename='circuit.gv')
        z = output_node
        tree_list = [z]
        z.build_tree(tree_list, DG, GDG)

        results_rows = []
        for row in self.rows:
            expected_out = z.compute_node(row)
            # uncomment the following line to see expected outputs for every possible combination
            # print(row, "----->", expected_out)
            row_result = []
            for node in tree_list:
                # force 0
                node.force_value = 0
                node_0_out = int(z.compute_node(row) != expected_out)
   
                # force 1
                node.force_value = 1
                node_1_out = int(z.compute_node(row) != expected_out)
                
                # unforce value
                node.force_value = None

                row_result.extend([node_0_out, node_1_out])
        
            results_rows.append(row_result)

        # for r in results_rows:
            # print(r)

        matrix = np.array(results_rows)
        final_result_set = set()

        # Find columns with exactly one '1'
        single_one_columns = [col for col in range(matrix.shape[1]) if np.sum(matrix[:, col] == 1) == 1]
        
        covered_col_idx = set()
        # Get the row indices for these columns
        row_indices = []
        for col in single_one_columns:
            row_index = np.where(matrix[:, col] == 1)[0][0]
            col_indices = np.where(matrix[row_index] == 1)[0] # of data type list
            set.update(covered_col_idx, col_indices)
            final_result_set.add(row_index)
            row_indices.append(row_index)

        matrix_col_indices = [i for i in range(matrix.shape[1])]
        matrix = matrix[:, list(set(matrix_col_indices) - covered_col_idx)]
        while True:
            
            sums = np.sum(matrix, axis=1)
            max_row_idx = np.argmax(sums)
            if sums[max_row_idx] == 0:
                break
            final_result_set.add(max_row_idx)
            col_indices = set(np.where(matrix[max_row_idx] == 1)[0])
            matrix_col_indices = [i for i in range(matrix.shape[1])]
            matrix = matrix[:, list(set(matrix_col_indices) - col_indices)]


        print("Test set: ", final_result_set)
        print(f'Test coverage = {100 * len(final_result_set)/len(self.rows):.2f}%')
        
        GDG.view()
        # self.draw_graph(DG) 

    def create_input_values(self):
        """
        n: number of inputs
        """
        n = self.num_input
        num_rows = 2 ** n
        rows = [[0 for j in range(n)] for i in range(num_rows)]

        for col_idx in range(n):
            num_alternate = 2 ** (n - col_idx - 1)  # Number of alternating blocks of 0s and 1s
            num_iter = int(num_rows / num_alternate)
            
            # Fill the rows for the current column with alternating blocks of 0s and 1s
            curr_bool = 0
            for z in range(num_iter):
                start = z * num_alternate
                end = (z + 1) * num_alternate
                for row in rows[start:end]:
                    row[col_idx] = curr_bool
                curr_bool = int(not bool(curr_bool))  # Toggle between 0 and 1
        return rows
    
    def setup(self, n):
        self.num_input = n
        self.rows = self.create_input_values()
        x = [Input(i) for i in range(self.num_input)]
        return x


if __name__ == '__main__':
    tester = Test()
    tester.test_function()