class Gate:
    # All gates should be listed here for graphviz(the graphing tool) to work properly
    gate_track = {'AND':0, 'OR':0, 'NOT':0, 'NAND':0, 'NOR':0, 'XOR':0, 'XNOR':0, 'x':0}
    used_names = []
    edges = []
    def __init__(self, input, force_value=None, name=None):
        try:
            self.name = name + str(self.gate_track[name])
            self.gate_track[name] += 1
        except:
            if name in self.used_names:
                new_name = name + str('#')
                self.name = new_name
                self.used_names.append(new_name)
            else:
                self.name = name
                self.used_names.append(name)
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
    
    def build_tree(self, tree_list, DG, GDG):
        """
        Builds the computation tree of the circuit.
        DG and GDG are just for the visualization tools
        """
        for node in self.input:
            if node not in tree_list:
                tree_list.append(node)
            DG.add_edge(node, self)
            if node.name + self.name in self.edges:
                pass
            else:
                self.edges.append(node.name + self.name)
                GDG.edge(node.name, self.name)

        for node in self.input:
            node.build_tree(tree_list, DG, GDG)
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.name

class And(Gate):
    def __init__(self, input, force_value=None, name="AND"):
        assert len(input) >= 2, f"And Gate expects 2 or more input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)

    def operation(self, *xs):
        """
        For an And gate to input 1, all the input must be 1, so if any input
        is 0, it should return 0.
        """
        p = 1
        for x in xs:
            if x == 0:
                p = 0
                break
        return min(1, p)

class Nand(Gate):
    def __init__(self, input, force_value=None, name="NAND"):
        assert len(input) >= 2, f"Nand Gate expects 2 or more input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)

    def operation(self, *xs):
        """
        A Nand gate is a replica of an And gate, with a negated return.
        """
        p = 1
        for x in xs:
            if x == 0:
                p = 0
                break
        return int(not bool(min(1, p)))

class Or(Gate):
    def __init__(self, input, force_value=None, name="OR"):
        assert len(input) >= 2, f"Or Gate expects 2 or more input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)
    
    def operation(self, *xs):
        """
        An Or gate evaluates to 1 if any of the input is 1
        Takes at least 2 inputs
        """
        s = 0
        for x in xs:
            if x == 1:
                s = 1
                break
        return min(1, s)

class Nor(Gate):
    def __init__(self, input, force_value=None, name="NOR"):
        assert len(input) >= 2, f"Nor Gate expects 2 or more input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)
    
    def operation(self, *xs):
        """
        A Nor gate is the replica of an Or gate with a negated return.
        Takes at least 2 inputs
        """
        s = 0
        for x in xs:
            if x == 1:
                s = 1
                break
        return int(not bool(min(1, s)))
    
class Not(Gate):
    def __init__(self, input, force_value=None, name="NOT"):
        assert len(input) == 1, f"Not Gate expects 1 input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)
    
    def operation(self, x):
        """
        A Not gate just returns the opposite of the truth value of its input
        """
        return int(not(x))
    
class Xor(Gate):
    def __init__(self, input, force_value=None, name="XOR"):
        assert len(input) >= 2, f"Xor Gate expects 2 or more input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)

    def operation(self, *xs):
        odd_count = sum(xs) % 2  # Count of '1's, mod 2 to check if odd
        return odd_count

class Xnor(Gate):
    def __init__(self, input, force_value=None, name="XNOR"):
        assert len(input) >= 2, f"Xnor Gate expects 2 or more input but {len(input)} were given"
        super().__init__(input=input, force_value=force_value, name=name)

    def operation(self, *xs):
        odd_count = sum(xs) % 2  # Count of '1's, mod 2 to check if odd
        return int(not bool(odd_count))

class Input(Gate):
    """
    These are leaf nodes of our tree
    """
    def __init__(self, col_id, force_value=None):
        assert type(col_id) == int, f"Input class expects an integer input but {type(col_id)} was given"
        super().__init__(input=col_id, force_value=force_value, name='x')
    
    def compute_node(self, row):
        if self.force_value != None:
            return self.force_value
        else:
            return row[self.input]
        
    def build_tree(self, *l):
        pass # has no children    

class Output(Gate):
    """
    This the root node of three, it expects a list of outputs from the circuit.
    Its value should never be forced, because it is not part of the circuit.
    """
    def __init__(self, input, force_value=None):
        super().__init__(input=input, force_value=force_value, name='output')       

    def compute_node(self, row):
        """
        Recursively computes the output at each node by going bottom up.
        Even though the calls are top down
        """
        if self.force_value != None:
            raise ValueError("Output node does not expect value to be forced as it is not part of the circuit")
        else:
            node_ouputs = []
            for node in self.input:
                node_ouputs.append(node.compute_node(row))
            return node_ouputs