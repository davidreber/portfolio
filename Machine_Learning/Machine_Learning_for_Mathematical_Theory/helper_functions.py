import numpy as np
import pandas as pd
from numpy import linalg as la
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from matplotlib import rcParams
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import time
from sklearn.svm import SVC

############################ GENERATE DATA ################################

def process_csv(csvfile):
    "Loads csv data. Returns as ndarray."
    networks = []
    with open(csvfile) as f:
        lines = f.readlines()
        # Each element (line) in lines is now a single network. 
        # Transform line to a ndarray, and append it to networks.
        for line in lines:
            # Divide up rows within each network
            rows = [item for item in line.strip().split('","')]
            # Remove starting and trailing braces
            rows[0] = rows[0][1:]
            rows[-1] = rows[-1][:-1]
            # Remove quotes around each row, split each entry up
            no_quotes = [item[1:-1].split(', ') for item in rows]
            array = np.array(no_quotes).astype(float)
            networks.append(array)
    # Convert all networks to a single 3-d array for storage.
    networks = np.array(networks)
    return networks

def one_hot(i):
    "Helper function for permute_generator. Creates a one-hot vector given a nonnegative integer."
    a = np.zeros(n)
    a[i] = 1
    return a

def permute_generator(n):
    "Yields all permutation matrices for dimension n."
    for ordering in itertools.permutations(range(n), n):
        P = np.zeros((n,n))
        for i in ordering:
            P[i,:] = one_hot(ordering[i], n)
        yield P
    
def remove_isomorphisms(network_list):
    "Returns a list of the nonisomorphic networks in network_list."
    keepers = []
    toremove = set()
    n = np.shape(nework_list[0])[0]
    for i, A in enumerate(network_list):
        if i in toremove:
            continue
        if i == len(network_list)-1: # last element reached, none to compare against
            continue
        for j, B in enumerate(network_list[i+1:]): 
            if i+j+1 in toremove:
                continue
            for P in permute_generator(n):
                if np.allclose(A, P.T.dot(B.dot(P))):
                    toremove.add(i+j+1)
                    break
    nonisomorphic = np.delete(network_list,sorted(toremove),axis=0)
    return list(nonisomorphic)

def add_self_nodes(data):
    "Add the self edges that Mathematica leaves out, and remove resulting isomorphisms. Returns new array."
    withself_data = []
    for i, network in enumerate(data):
        # For each network, generate permutations of self-loops
        selfs_per_network = []
        self_edges = itertools.product([0,1], repeat = n)
        for selfs in self_edges:
            selfs_per_network.append(network + np.diag(selfs))
            
        # Remove any ismorphisms generated
        cleaned_per_network = remove_isomorphisms(selfs_per_network)
        
        # Add the new nonisomorphic networks to the new dataset
        withself_data.extend(cleaned_per_network)
    return np.array(withself_data)

def radius(A): 
    "Return the spectral radius of A."
    return la.norm(la.eigvals(A), np.inf)

def delay(A,entry):
    """Time-delays a network in the designated edge by 1 time-step.
    Accepts: 
        A (ndarray): a single network to time delay
        entry (2-tuple): indicies of which entry to time delay
    Returns:
        A_d (ndarray): the time delayed version of A"""
    i,j = entry[0],entry[1]
    A_d = np.zeros((len(A)+1,len(A)+1))
    A_d[:-1,:-1] = A
    A_d[i,-1] = A[i,j]
    A_d[-1,j] = 1.
    A_d[i,j] = 0.
    return A_d

def tabular_effects(data, fill=None):
    """Accepts: 
           data (a list of networks).
       Returns: 
           effects (ndarray): has shape like data , where each
               k-i-j element is the spectral radius obtained by delaying 
               the k'th network in the i-j'th entry"""
    effects = np.zeros_like(data)
    base_radii = []
    for k, network in enumerate(data):
        base_radius = radius(network)
        base_radii.append(base_radius)
        for i, row in enumerate(network):
            for j, val in enumerate(row):
                if val == 0.:
                    effects[k,i,j] = fill
                else:
                    r_delay = radius(delay(network,(i,j)))
                    effects[k,i,j] = r_delay - base_radius
    return effects, np.array(base_radii)

############################ FEATURE GENERATION ################################

def find_cycles(network, fulledge=True):
    """Finds all the cycles in a network. Returns a list of cycles, 
    each cycle being a list of edges (which are tuples)."""  
    new = np.copy(network)
    n = int(np.sqrt(np.product(np.shape(new))))
    adjacency = np.reshape(new,(n,n)).astype(float)
    D = nx.DiGraph(adjacency)
    edge_cycles = []
    for vertex_cycle in list(nx.simple_cycles(D)):
        if len(vertex_cycle) == 1:
            edge_cycles.append([tuple(vertex_cycle + vertex_cycle),])
        else:
            mod_vertices = vertex_cycle + vertex_cycle[:1]
            cycle = [(item,mod_vertices[i+1]) for i, item in enumerate(mod_vertices[:-1])]
            edge_cycles.append(cycle)
    if fulledge:
        return edge_cycles
    else:
        possible_edges = sorted([entry for entry in permutations(range(n),2)] + zip(range(n),range(n)))
        shorthand_edges = []
        for cycle in edge_cycles:
            shorthand = [possible_edges.index(edge) for edge in cycle]
            shorthand_edges.append(shorthand)
        return shorthand_edges
    
def get_cycle_products(network, cycles):
    """Computes the cycle products in a network. Returns a list of cycle products."""  
    return [np.product([network[entry] for entry in cycle]) for cycle in cycles]

def edge_find_cycles(edge,network): 
    "Finds cycles in a network corresponding to a given edge"
    return [cycle for cycle in find_cycles(network) if edge in cycle]

def feature_gen_all(networks, effects):
    """Finds all cycles, in all the networks in the dataset, and returns data of interest.
    Returns:
    """
    dim = np.shape(networks[0])[0]
    X = []
    y = []
    for n, network in enumerate(networks):
        effect = effects[n]
        evals = list(np.sort(np.abs(np.linalg.eigvals(network))))[::-1]
        for i in xrange(dim):
            for j in xrange(dim):
                edge = (i,j)
                if network[edge]:
                    row = features_row(network, effect, edge, dim)
                    row.extend(evals)
                    X.append(row)
                    y.append(effect[edge])
    headers = ['Weight','Num_Cycles','Cycle_Prod_Sum','Abs_Cycle_Prod_Sum']
    headers += ['Num_'+str(k)+'_Cycles' for k in xrange(1,dim+1)]
    headers += ['Sum_'+str(k)+'_Products' for k in xrange(1,dim+1)]
    headers += ['Sum_'+str(k)+'_Abs_Prod' for k in xrange(1,dim+1)]
    headers.append('RIS_Criteria_Val')
    headers += ['det_slope','det_intercept','val*det_slope','slope/intecept','slope-intercept','abs(slope-intercept)']
    headers += ['Eigenvalue_'+str(k)+'_Norm' for k in xrange(1,dim+1)]
    data = pd.DataFrame(np.array(X), columns=headers)
    labels = pd.Series(y)
    return data, labels

def features_row(network, effect, edge, dim):
    row = []
    cycles = edge_find_cycles(edge,network)
    cycle_products = get_cycle_products(network, cycles)
    len_cycles = [len(cycle) for cycle in cycles]
    num_lenk_cycles = [len_cycles.count(k) for k in xrange(1,dim+1)]
    # Divide out cycle products by the length of their cycle
    cycle_products_by_lenk = [[]]*dim
    for i, product in enumerate(cycle_products):
        cycle_products_by_lenk[len_cycles[i]-1].append(product) # TODO: verify works!
    sum_products_lenk = [np.sum(k_cycles) for k_cycles in cycle_products_by_lenk]
    sum_abs_products_lenk = [np.sum(np.abs(k_cycles)) for k_cycles in cycle_products_by_lenk]
    RIS_val = np.sum(np.abs(sum_products_lenk))
    
    # Append features
    row.append(network[edge])
    row.append(len(cycles))
    row.append(np.sum(cycle_products))
    row.append(np.sum(np.abs(cycle_products)))
    row.extend(num_lenk_cycles)
    row.extend(sum_products_lenk)
    row.extend(sum_abs_products_lenk)
    row.append(RIS_val)
    row.extend(determinant_features(network,edge))
    return row

def lenk_cycle_product_sums(networks, effects):
    cycle_product_sums_lenk = dict()
    delta_rho_lenk = dict()
    for n, network in enumerate(networks):
        effect = effects[n]
        for i in xrange(np.shape(network)[0]):
            for j in xrange(np.shape(network)[1]):
                if network[(i,j)]:
                    cycles = edge_find_cycles((i,j),network)
                    for cycle in cycles:
                        if len(cycle) in cycle_product_sums_lenk:
                            cycle_product_sums_lenk[len(cycle)].append(np.product([network[edge] for edge in cycle]))
                            delta_rho_lenk[len(cycle)].append(effect[(i,j)])
                        else:
                            cycle_product_sums_lenk[len(cycle)] = [np.product([network[edge] for edge in cycle]),]
                            delta_rho_lenk[len(cycle)] = [effect[(i,j)],]
                                        
    return cycle_product_sums_lenk, delta_rho_lenk

def det_prod_3(network, entry):
    """Computes det_slope and det_intercept, found by factoring the determinant 
    of a 3x3 network as follows:
            determinat(network) = det_slope * network[entry] + det_intercept
    Accepts:
        network: an ndarray of shape (3,3)
        entry: a tuple of form (i,j)
    Returns:
        det_slope: scalar, as defined above
        det_intercept: scalar, as defined above
        """
    a11, a12, a13 = network[0]
    a21, a22, a23 = network[1]
    a31, a32, a33 = network[2]
    cycle_prods = {-a13*a22*a31:[(0,2),(1,1),(2,0)],
                   a12*a23*a31: [(0,1),(1,2),(2,0)],
                   a13*a21*a32: [(0,2),(1,0),(2,1)],
                   -a11*a23*a32:[(0,0),(1,2),(2,1)],
                   -a12*a21*a33:[(0,1),(1,0),(2,2)],
                   a11*a22*a33: [(0,0),(1,1),(2,2)]}
    det_slope = 0
    det_intercept = 0
    for prod in cycle_prods.keys():
        if entry in cycle_prods[prod]:
            det_slope += prod
        else:
            det_intercept += prod
    return det_slope, det_intercept

def det_prod_4(network, entry):
    """Computes det_slope and det_intercept, found by factoring the determinant 
    of a 4x4 network as follows:
            determinat(network) = det_slope * network[entry] + det_intercept
    Accepts:
        network: an ndarray of shape (4,4)
        entry: a tuple of form (i,j)
    Returns:
        det_slope: scalar, as defined above
        det_intercept: scalar, as defined above
        """
    a11, a12, a13, a14 = network[0]
    a21, a22, a23, a24 = network[1]
    a31, a32, a33, a34 = network[2]
    a41, a42, a43, a44 = network[3]
    cycle_prods = {a11*a22*a33*a44:[(0, 0), (1, 1), (2, 2), (3, 3)],
                   -a11*a22*a34*a43:[(0, 0), (1, 1), (2, 3), (3, 2)],
                   -a11*a23*a32*a44:[(0, 0), (1, 2), (2, 1), (3, 3)],
                   a11*a23*a34*a42:[(0, 0), (1, 2), (2, 3), (3, 1)],
                   a11*a24*a32*a43:[(0, 0), (1, 3), (2, 1), (3, 2)],
                   -a11*a24*a33*a42:[(0, 0), (1, 3), (2, 2), (3, 1)],
                   -a12*a21*a33*a44:[(0, 1), (1, 0), (2, 2), (3, 3)],
                   a12*a21*a34*a43:[(0, 1), (1, 0), (2, 3), (3, 2)],
                   a12*a23*a31*a44:[(0, 1), (1, 2), (2, 0), (3, 3)],
                   -a12*a23*a34*a41:[(0, 1), (1, 2), (2, 3), (3, 0)],
                   -a12*a24*a31*a43:[(0, 1), (1, 3), (2, 0), (3, 2)],
                   a12*a24*a33*a41:[(0, 1), (1, 3), (2, 2), (3, 0)],
                   a13*a21*a32*a44:[(0, 2), (1, 0), (2, 1), (3, 3)],
                   -a13*a21*a34*a42:[(0, 2), (1, 0), (2, 3), (3, 1)],
                   -a13*a22*a31*a44:[(0, 2), (1, 1), (2, 0), (3, 3)],
                   a13*a22*a34*a41:[(0, 2), (1, 1), (2, 3), (3, 0)],
                   a13*a24*a31*a42:[(0, 2), (1, 3), (2, 0), (3, 1)],
                   -a13*a24*a32*a41:[(0, 2), (1, 3), (2, 1), (3, 0)],
                   -a14*a21*a32*a43:[(0, 3), (1, 0), (2, 1), (3, 2)],
                   a14*a21*a33*a42:[(0, 3), (1, 0), (2, 2), (3, 1)],
                   a14*a22*a31*a43:[(0, 3), (1, 1), (2, 0), (3, 2)],
                   -a14*a22*a33*a41:[(0, 3), (1, 1), (2, 2), (3, 0)],
                   -a14*a23*a31*a42:[(0, 3), (1, 2), (2, 0), (3, 1)],
                   a14*a23*a32*a41:[(0, 3), (1, 2), (2, 1), (3, 0)]}
    det_slope = 0
    det_intercept = 0
    for prod in cycle_prods.keys():
        if entry in cycle_prods[prod]:
            det_slope += prod
        else:
            det_intercept += prod
    return det_slope, det_intercept

def determinant_features(network, entry):
    dim = np.shape(network)[0]
    if dim == 3:
        det_slope, det_intercept = det_prod_3(network, entry)
    elif dim == 4:
        det_slope, det_intercept = det_prod_4(network, entry)
    else:
        raise ValueError("Functionality not yet added for dim="+str(dim))
        
    val = network[entry]
    toret = [det_slope, det_intercept, val*det_slope]
    toret.append(val*det_slope/(det_intercept+1e-1))
    toret.append(val*det_slope - det_intercept)
    toret.append(np.abs(val*det_slope - det_intercept))
    return toret

############################ DATA VISUALIZATION ################################

def cycle_values(networks):
    """Finds all cycle edge values, in all the networks in the dataset. 
    Returns a list (the networks) of lists of lists of tuples (entries)"""
    return [find_cycles(network) for network in networks]

def all_cycles_vals_effects(networks, effects):
    """Finds all cycles, in all the networks in the dataset, and returns data of interest.
    Returns:
        all_cycles: list (all networks) of lists (a specific network) of lists (a cycle) of tuples (entries)
        all_cycles_vals: the corresponding edge values for all of the edges in all_cycles
        all_cycles_effects: the corresponding effect values for all of the edges in all_cycles
    """
    all_cycles = cycle_values(networks)
    # create a corresponding dataset of the entries of the matrix
    all_cycles_edges = []
    all_cycles_effects = []
    for i, net_cycles in enumerate(all_cycles):
        net_edges = []
        net_effects = []
        network = networks[i]
        effect = effects[i]
        for cycle in net_cycles:
            cycle_edges= []
            cycle_effects = []
            for entry in cycle:
                cycle_edges.append(network[entry])
                cycle_effects.append(effect[entry])
            net_edges.append(cycle_edges)
            net_effects.append(cycle_effects)
        all_cycles_edges.append(net_edges)
        all_cycles_effects.append(net_effects)
    return all_cycles, all_cycles_edges, all_cycles_effects

def ravel_on_cycle(cycles, cycles_edges, cycles_effects):
    """Ravels the aggregate sublist inputs of all_cycles, all_cycles_edges, all_cycles_effects 
    to a list of cycle-based items."""
    
    def ravel_single(aggregate):
        "Ravels a single aggregate item"
        cycles = []
        for network_cycles in aggregate:
            cycles.extend(network_cycles)
        return cycles
    
    return ravel_single(cycles), ravel_single(cycles_edges), ravel_single(cycles_effects)

def violin_plot(frame,col,title,xlabel,ylabel): 
    "Makes a violin plot of a pandas DataFrame with discrete independent variable"
    grouped = frame.groupby(col)
    pos = grouped.groups.keys()
    data = [grouped.get_group(key)[1].as_matrix() for key in grouped.groups.keys()]
    plt.violinplot(data,pos,showextrema=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
############################ MACHINE LEARNING ################################
    
class Learner(object):
    """Runs several machine learning algorithms on a set of data"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def learn_linear(self, X, y, num_runs, train_split=0.7):
        R2 = []
        weights = []
        for i in xrange(num_runs):
            train_x, test_x, train_y, test_y = train_test_split(X, y,train_size=train_split)
            linear = lm.LinearRegression()
            linear.fit(train_x, train_y)
            R2.append(linear.score(test_x,test_y))
            weights.append(list(linear.coef_))
        weights2 = np.mean(np.array(weights), axis=0)
        self.linear_coef = weights2
        self.linear_acc = np.mean(R2)
        self.linear_std = np.std(R2)
        
    def learn_ridge(self, X, y, num_runs, train_split=0.7):
        R2 = []
        weights = []
        for i in xrange(num_runs):
            train_x, test_x, train_y, test_y = train_test_split(X, y,train_size=train_split)
            ridge = lm.Ridge()
            ridge.fit(train_x, train_y)
            R2.append(ridge.score(test_x,test_y))
            weights.append(list(ridge.coef_))
        weights2 = np.mean(np.array(weights), axis=0)
        self.ridge_coef = weights2
        self.ridge_acc = np.mean(R2)
        self.ridge_std = np.std(R2)
        
    def learn_logistic(self, X, y_positive, num_runs, train_split=0.7):
        accs = []
        weights = []
        for i in xrange(num_runs):
            train_x, test_x, train_y, test_y = train_test_split(X, y_positive,train_size=train_split)
            logistic = lm.LogisticRegression()
            logistic.fit(train_x, train_y)
            accs.append(logistic.score(test_x,test_y))
            weights.append(list(logistic.coef_))
        weights2 = np.mean(np.array(weights), axis=0)
        self.logistic_coef = weights2
        self.logistic_acc = np.mean(accs)
        self.logistic_std = np.std(accs)
        
    def learn_tree(self, X, y_positive, num_runs, train_split=0.7):
        accs = []
        weights = []
        for i in xrange(num_runs):
            train_x, test_x, train_y, test_y = train_test_split(X, y_positive,train_size=train_split)
            tree = DecisionTreeClassifier(min_samples_leaf=10) # TODO: tune parameters more ?
            tree.fit(train_x, train_y)
            accs.append(tree.score(test_x,test_y))
            weights.append(list(tree.feature_importances_))
        weights2 = np.mean(np.array(weights), axis=0)
        self.tree_coef = weights2
        self.tree_acc = np.mean(accs)
        self.tree_std = np.std(accs)
        
    def learn_forest_regression(self, X, y, num_runs, train_split=0.7):
        R2 = []
        weights = []
        for i in xrange(num_runs):
            train_x, test_x, train_y, test_y = train_test_split(X, y,train_size=train_split)
            forest = RandomForestRegressor()
            forest.fit(train_x, train_y)
            R2.append(forest.score(test_x,test_y))
            weights.append(list(forest.feature_importances_))
        weights2 = np.mean(np.array(weights), axis=0)
        self.forest_reg_coef = weights2
        self.forest_reg_acc = np.mean(R2)
        self.forest_reg_std = np.std(R2)
        
    def learn_forest_class(self, X, y_positive, num_runs, train_split=0.7):
        accs = []
        weights = []
        for i in xrange(num_runs):
            train_x, test_x, train_y, test_y = train_test_split(X, y_positive,train_size=train_split)
            forest = RandomForestClassifier()
            forest.fit(train_x, train_y)
            accs.append(forest.score(test_x,test_y))
            weights.append(list(forest.feature_importances_))
        weights2 = np.mean(np.array(weights), axis=0)
        self.forest_class_coef = weights2
        self.forest_class_acc = np.mean(accs)
        self.forest_class_std = np.std(accs)
    
    def learn_all(self,train_split=0.7,N_linear=100,N_ridge=100,N_logit=100,
                  N_tree=5,N_forest_reg=5,N_forest_class=5):
        X = self.X
        y = self.y
        y_positive = y > 0
        
        # Train and score all the methods
        self.learn_linear(X, y, N_linear, train_split)
        self.learn_ridge(X, y, N_ridge, train_split)
        self.learn_logistic(X, y_positive, N_logit, train_split)
        self.learn_tree(X, y_positive, N_tree, train_split)
        self.learn_forest_regression(X, y, N_forest_reg, train_split)
        self.learn_forest_class(X, y_positive, N_forest_class, train_split)
        
        # Organize results to return
        accs = [self.linear_acc,self.ridge_acc,self.logistic_acc,self.tree_acc,
                self.forest_reg_acc,self.forest_class_acc]
        stds = [self.linear_std,self.ridge_std,self.logistic_std,self.tree_std,
                self.forest_reg_std,self.forest_class_std]
        coefs = [self.linear_coef,self.ridge_coef,self.logistic_coef,self.tree_coef,
                 self.forest_reg_coef,self.forest_class_coef]
        
        return np.array(accs), np.array(stds), np.array(coefs)
    