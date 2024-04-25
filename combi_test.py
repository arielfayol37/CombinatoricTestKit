import numpy as np

def And(*xs):
    p = 1
    for x in xs:
        if x == 0:
            p = 0
            break
    return min(1, p)

def Not(x):
    return int(not(x))

def Or(*xs):
    s = 0
    for x in xs:
        if x == 1:
            s = 1
            break
    return min(1, s)

def Nor(*xs):
    return int(not bool(Or(*xs)))

def Nand(*xs):
    return int(not bool(And(*xs)))
    

class Gate:
    def __init__(self, input, operation, force_value=None, name=None):
        self.name = name
        self.operation = operation
        self.input = input
        self.force_value = force_value

    def compute_node(self, row):
        """
        Recursively computes the output at each node by going bottom up.
        Even though the calls are top down
        """
        if self.force_value != None:
            return self.force_value
        else:
            node_ouputs = []
            for node in self.input:
                node_ouputs.append(node.compute_node(row))
            return self.operation(*node_ouputs)
    
    def build_tree(self, tree_list):
        for node in self.input:
            if node not in tree_list:
                tree_list.append(node)
        for node in self.input:
            node.build_tree(tree_list)
    
    def __str__(self) -> str:
        return self.name

class Input(Gate):
    """
    These are leaf nodes of our tree
    """
    def __init__(self, col_id, force_value=None):
        super().__init__(input=col_id, operation=None, force_value=force_value, name=chr(97 + col_id))
        self.input = col_id
    
    def compute_node(self, row):
        if self.force_value != None:
            return self.force_value
        else:
            return row[self.input]
        
    def build_tree(self, tree_list):
        pass # has no children    

class Test:
    def __init__(self, name=None):
        self.name = name

    def test_function(self):
        """
        
        ------THIS IS THE WHERE THE USER IS EXPECTED TO INPUT THEIR FUNCTION
        
        """
        self.num_input = 5
        self.rows = self.create_input_values()
        x = [Input(i) for i in range(self.num_input)]
        and0 = Gate([x[0], x[1]], And, name='AND0')
        and1 = Gate([x[2], x[3]], And, name='AND1')
        not0 = Gate([and1], Not, name="NOT0")
        or0 = Gate([not0, and0], Or, name="OR0")
        or1 = Gate([and1, x[4]], Or, name="OR1")
        z = Gate([or0, or1], And, name='Z')
        """
        ------- END OF USER INPUT
        """

        tree_list = [z]
        z.build_tree(tree_list)
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


        print("Test set ", final_result_set)

        print(f'Test coverage = {100 * len(final_result_set)/len(self.rows):.2f}%')
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




    
def test_base_functions():
    # Testing And method
    print("Testing And method")
    for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"And({x}, {y}) = {And(x, y)}")

    # Testing Or method
    print("Testing Or method")
    for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"Or({x}, {y}) = {Or(x, y)}")
    
    # Testing Nor method
    print("Testing Nor method")
    for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"Nor({x}, {y}) = {Nor(x, y)}")

    # Testing Nand method
    print("Testing Nand method")
    for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        print(f"Nand({x}, {y}) = {Nand(x, y)}")

    # Testing Not method
    print("Testing Not method")
    for x in [0, 1]:
        print(f"Not({x}) = {Not(x)}")

# Call the test function
# test_base_functions()

tester = Test()
tester.test_function()