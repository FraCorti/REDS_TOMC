import math
import os
from functools import partial

from ortools.linear_solver import pywraplp
from utils.logs import log_print
import time


def setup_gurobi(args):
    os.environ[
        'GUROBI_HOME'] = args.gurobi_home
    os.environ['GRB_LICENSE_FILE'] = args.gurobi_license_file

    print("Gurobi settings:")
    print(os.getenv('GUROBI_HOME'))
    print(os.getenv('GRB_LICENSE_FILE'))


def knapsack_find_splits_mobilenetv1(importance_score_feature_extraction_filters, macs_list, memory_list, macs_targets,
                                     memory_targets,
                                     importance_score_pointwise_filters_kernels,
                                     layers_filter_macs, args, bottom_up=True,
                                     last_pointwise_filters=1) \
        :
    feature_extraction_layer_filters = [list(range(0, len(importance_score_feature_extraction_filters[i]), 1)) for i in
                                        range(len(importance_score_feature_extraction_filters))]

    dense_model_macs = [sum(layers_filter_macs[layer_index]) for layer_index in range(len(layers_filter_macs))]
    subnetworks_macs = [int(sum(dense_model_macs))]

    subnetworks_filters_first_convolution = [[] for _ in range(args.subnets_number - 1)]
    subnetworks_filters_pointwise = [[] for _ in range(args.subnets_number - 1)]
    subnetworks_filters_depthwise = [[] for _ in range(args.subnets_number - 1)]

    if bottom_up:
        log_print("Running bottom-up solver")
        # initialize subsolutions data structure to store the optimal solutions for each subnetwork found by the solver
        optimal_subsolutions_first_layer = [0 for _ in range(len(importance_score_feature_extraction_filters[0]))]

        optimal_subsolutions_feature_extraction = [
            [0 for _ in range(len(importance_score_feature_extraction_filters[i]))] for i in
            range(1, len(importance_score_feature_extraction_filters))]
        optimal_subsolutions_feature_creation = [[0 for _ in range(len(importance_score_pointwise_filters_kernels[i]))]
                                                 for i in
                                                 range(len(importance_score_pointwise_filters_kernels))]
    else:
        log_print("Running top-down solver")
        macs_targets.reverse()

    residual_macs = 0

    for subnetwork_mac_index in range(len(macs_targets)):

        subnetwork_residual_macs = -1
        solver_time_multiplier = 1
        solver_max_iteration = args.solver_max_iterations

        while subnetwork_residual_macs <= 0 and solver_max_iteration > 0:

            log_print("Solver iteration: {} MACs capacity: {}".format(solver_time_multiplier, int(
                macs_targets[subnetwork_mac_index] + residual_macs)))

            if bottom_up:
                first_layer_index, depthwise_layers_indexes, pointwise_layers_indexes, solution_macs = ortools_knapsack_solver_mobilenetv1(
                    feature_extraction_layers_filters_indexes=feature_extraction_layer_filters,
                    weights_macs_layers=macs_list,
                    weights_memory=memory_list,
                    feature_extraction_filters_score=importance_score_feature_extraction_filters,
                    capacity_macs=int(macs_targets[subnetwork_mac_index] + residual_macs) if bottom_up else int(
                        macs_targets[subnetwork_mac_index]),
                    importance_score_pointwise_filters_kernels=importance_score_pointwise_filters_kernels,
                    capacity_memory_size=memory_targets[subnetwork_mac_index],
                    previous_optimal_solution_first_layer=optimal_subsolutions_first_layer,
                    previous_optimal_solution_feature_extraction=optimal_subsolutions_feature_extraction,
                    previous_optimal_solution_feature_creation=optimal_subsolutions_feature_creation,
                    solver_time_limit=args.solver_time_limit * solver_time_multiplier,
                    last_pointwise_filters=last_pointwise_filters,
                    last_solution_iteration_search=True if solver_time_multiplier == solver_max_iteration else False)

            else:
                first_layer_index, depthwise_layers_indexes, pointwise_layers_indexes, solution_macs = ortools_knapsack_solver_mobilenetv1(
                    feature_extraction_layers_filters_indexes=feature_extraction_layer_filters,
                    weights_macs_layers=macs_list,
                    weights_memory=memory_list,
                    importance_score_pointwise_filters_kernels=importance_score_pointwise_filters_kernels,
                    feature_extraction_filters_score=importance_score_feature_extraction_filters,
                    capacity_macs=int(macs_targets[subnetwork_mac_index] + residual_macs) if bottom_up else int(
                        macs_targets[subnetwork_mac_index]),
                    capacity_memory_size=memory_targets[subnetwork_mac_index],
                    solver_time_limit=args.solver_time_limit * solver_time_multiplier,
                    last_pointwise_filters=last_pointwise_filters,
                    last_solution_iteration_search=True if solver_time_multiplier == solver_max_iteration else False)

            subnetwork_residual_macs = macs_targets[subnetwork_mac_index] + residual_macs - solution_macs

            if subnetwork_residual_macs >= 0:

                log_print("The solver found a solution. Search the next solution.")

                # store the optimal solution for the current subnetwork
                subnetworks_filters_first_convolution[subnetwork_mac_index].append(first_layer_index)
                subnetworks_filters_depthwise[subnetwork_mac_index].append(depthwise_layers_indexes)
                subnetworks_filters_pointwise[subnetwork_mac_index].append(pointwise_layers_indexes)
                subnetworks_macs.append(solution_macs)

                residual_macs = subnetwork_residual_macs  # update the residual macs for the next iteration

                if residual_macs < 0:
                    residual_macs = 0
                else:
                    log_print("Residual macs: {}, added to the next iteration".format(residual_macs))

                break

            solver_time_multiplier += 1
            solver_max_iteration -= 1

            log_print("MACs target: {} Current network architecture MACS: {} Residual macs: {}".format(
                macs_targets[subnetwork_mac_index], solution_macs, residual_macs))

    return subnetworks_filters_first_convolution, subnetworks_filters_depthwise, subnetworks_filters_pointwise, subnetworks_macs


def knapsack_find_splits_ds_cnn(importance_list, macs_list, memory_list, macs_targets, memory_targets,
                                importance_score_pointwise_filters_kernels,
                                layers_filter_macs, units_layer_size, args, bottom_up=True, last_pointwise_filters=1,
                                model_size="s"):
    layers = [list(range(0, len(importance_list[i]), 1)) for i in range(len(importance_list))]

    dense_model_macs = [sum(layers_filter_macs[layer_index]) for layer_index in range(len(layers_filter_macs))]
    subnetworks_macs = [int(sum(dense_model_macs))]

    subnetworks_filters_first_convolution = [[] for _ in range(args.subnets_number - 1)]
    subnetworks_filters_pointwise = [[] for _ in range(args.subnets_number - 1)]
    subnetworks_filters_depthwise = [[] for _ in range(args.subnets_number - 1)]

    if bottom_up:
        log_print("Running bottom-up solver")
        # initialize subsolutions data structure to store the optimal solutions for each subnetwork found by the solver
        optimal_subsolutions_first_layer = [0 for _ in range(units_layer_size)]

        optimal_subsolutions_feature_extraction = [[0 for _ in range(len(importance_list[i]))] for i in
                                                   range(len(importance_list) - 1)]
        optimal_subsolutions_feature_creation = [[0 for _ in range(len(importance_list[i]))] for i in
                                                 range(len(importance_list) - 1)]
    else:
        log_print("Running top-down solver")
        macs_targets.reverse()

    residual_macs = 0

    for subnetwork_mac_index in range(len(macs_targets)):

        subnetwork_residual_macs = -1
        solver_time_multiplier = 1
        solver_max_iteration = args.solver_max_iterations

        while subnetwork_residual_macs <= 0 and solver_max_iteration > 0:

            log_print("Solver iteration: {} MACs capacity: {}".format(solver_time_multiplier, int(
                macs_targets[subnetwork_mac_index])))

            if bottom_up:
                first_layer_index, depthwise_layers_indexes, pointwise_layers_indexes, solution_macs = ortools_knapsack_solver_ds_cnn(
                    layers=layers, weights_macs_layers=macs_list,
                    weights_memory=memory_list,
                    filters_score=importance_list,
                    units_layer_size=units_layer_size,
                    capacity_macs=int(macs_targets[subnetwork_mac_index]) if bottom_up else int(
                        macs_targets[subnetwork_mac_index]),  # + residual_macs
                    importance_score_pointwise_filters_kernels=importance_score_pointwise_filters_kernels,
                    capacity_memory_size=memory_targets[subnetwork_mac_index],
                    previous_optimal_solution_first_layer=optimal_subsolutions_first_layer,
                    previous_optimal_solution_feature_extraction=optimal_subsolutions_feature_extraction,
                    previous_optimal_solution_feature_creation=optimal_subsolutions_feature_creation,
                    solver_time_limit=args.solver_time_limit * solver_time_multiplier,
                    last_pointwise_filters=last_pointwise_filters,
                    last_solution_iteration_search=True if solver_time_multiplier == solver_max_iteration else False,
                    model_size=model_size)
            else:
                first_layer_index, depthwise_layers_indexes, pointwise_layers_indexes, solution_macs = ortools_knapsack_solver_ds_cnn(
                    layers=layers, weights_macs_layers=macs_list,
                    weights_memory=memory_list,
                    importance_score_pointwise_filters_kernels=importance_score_pointwise_filters_kernels,
                    filters_score=importance_list,
                    units_layer_size=units_layer_size,
                    capacity_macs=int(macs_targets[subnetwork_mac_index]) if bottom_up else int(
                        macs_targets[subnetwork_mac_index]),  # + residual_macs
                    capacity_memory_size=memory_targets[subnetwork_mac_index],
                    solver_time_limit=args.solver_time_limit * solver_time_multiplier,
                    last_pointwise_filters=last_pointwise_filters,
                    last_solution_iteration_search=True if solver_time_multiplier == solver_max_iteration else False,
                    model_size=model_size)

            subnetwork_residual_macs = macs_targets[subnetwork_mac_index] + residual_macs - solution_macs

            if subnetwork_residual_macs >= 0:

                log_print("The solver found a solution. Search the next solution.")

                # store the optimal solution for the current subnetwork
                subnetworks_filters_first_convolution[subnetwork_mac_index].append(first_layer_index)
                subnetworks_filters_depthwise[subnetwork_mac_index].append(depthwise_layers_indexes)
                subnetworks_filters_pointwise[subnetwork_mac_index].append(pointwise_layers_indexes)
                subnetworks_macs.append(solution_macs)

                residual_macs = subnetwork_residual_macs  # update the residual macs for the next iteration

                if residual_macs < 0:
                    residual_macs = 0
                else:
                    log_print("Residual macs: {}, added to the next iteration".format(residual_macs))

                break

            solver_time_multiplier += 1
            solver_max_iteration -= 1

            log_print("MACs target: {} Current network architecture MACS: {} Residual macs: {}".format(
                macs_targets[subnetwork_mac_index], solution_macs, residual_macs))

    log_print("Subnetworks first convolution: {}".format(subnetworks_filters_first_convolution))
    log_print("Subnetworks filters depthwise: {}".format(subnetworks_filters_depthwise))
    log_print("Subnetworks filters pointwise: {}".format(subnetworks_filters_pointwise))

    return subnetworks_filters_first_convolution, subnetworks_filters_depthwise, subnetworks_filters_pointwise, subnetworks_macs


def knapsack_find_splits_dnn(importance_list, macs_list, memory_list, macs_targets, memory_targets, layers_filter_macs,
                             args, bottom_up=True):
    classes = [list(range(0, len(importance_list[i]), 1)) for i in range(len(importance_list))]

    if bottom_up:
        log_print("Running Bottom-up solver")
    else:
        log_print("Running Top-down solver")
        macs_targets.reverse()

    subnetworks_macs = [int(sum(layers_filter_macs[0]) + sum(layers_filter_macs[1] + sum(layers_filter_macs[2])))]
    for subnetwork_macs in macs_targets:
        subnetworks_macs.append(subnetwork_macs)

    subnetworks_neurons_indexes = [[] for _ in range(args.subnets_number - 1)]

    # initialize subsolutions data structure
    optimal_subsolutions = [[0 for _ in range(len(importance_list[i]))] for i in range(len(importance_list))]

    # apply the solver on the model to find the splits
    for subnetwork_mac_index in range(len(macs_targets)):
        filter_indexes = ortools_knapsack_solver_dnn(classes=classes, weights_macs=macs_list,
                                                     weights_memory=memory_list,
                                                     filters_score=importance_list,
                                                     capacity_macs=macs_targets[subnetwork_mac_index],
                                                     capacity_memory_size=memory_targets[subnetwork_mac_index],
                                                     bottom_up=bottom_up,
                                                     previous_optimal_solution=optimal_subsolutions if bottom_up else None)
        subnetworks_neurons_indexes[subnetwork_mac_index].append(filter_indexes)

    print("Subnetworks splittings: {}".format(subnetworks_neurons_indexes))

    return subnetworks_neurons_indexes, subnetworks_macs


def knapsack_find_splits_cnn(importance_list, macs_list, memory_list, macs_targets, memory_targets, layers_filter_macs,
                             args, bottom_up=True):
    classes = [list(range(0, len(importance_list[i]), 1)) for i in range(len(importance_list))]

    if bottom_up:
        log_print("Running Bottom-up solver")
    else:
        log_print("Running Top-down solver")
        macs_targets.reverse()

    subnetworks_macs = [int(sum(layers_filter_macs[0]) + sum(layers_filter_macs[1]))]
    for subnetwork_macs in macs_targets:
        subnetworks_macs.append(subnetwork_macs)

    subnetworks_filters_indexes = [[] for _ in range(args.subnets_number - 1)]

    contains_zero = (lambda f, nested_list: any(

        f(f, sub_list) if isinstance(sub_list, list) else sub_list == 0 for sub_list in nested_list))
    contains_zero_recursive = partial(contains_zero, contains_zero)

    correctness = False

    while correctness is False:

        # initialize subsolutions data structure
        optimal_subsolutions = [[0 for _ in range(len(importance_list[i]))] for i in range(len(importance_list))]

        # apply the solver on the model to find the splits
        for subnetwork_mac_index in range(len(macs_targets)):
            filter_indexes = ortools_knapsack_solver_cnn(classes=classes, weights_macs=macs_list,
                                                         weights_memory=memory_list,
                                                         filters_score=importance_list,
                                                         capacity_macs=macs_targets[subnetwork_mac_index],
                                                         capacity_memory_size=memory_targets[subnetwork_mac_index],
                                                         bottom_up=bottom_up,
                                                         previous_optimal_solution=optimal_subsolutions if bottom_up else None)
            subnetworks_filters_indexes[subnetwork_mac_index].append(filter_indexes)

        # check if all the layers contain at least one filter in each subnetwork
        if not contains_zero_recursive(subnetworks_filters_indexes):
            correctness = True
        else:
            subnetworks_filters_indexes = [[] for _ in range(args.subnets_number)]
            log_print("The solver found a subnetwork with a layer without filters. Restarting the solver...")

    print("Subnetworks splittings: {}".format(subnetworks_filters_indexes))

    return subnetworks_filters_indexes, subnetworks_macs


def initialize_nested_knapsack_solver_dnn(layers_filters_macs, layers_filters_byte,
                                          descending_importance_score_scores,
                                          int_scale_value=1e5, subnetworks_number=3, constraints_percentages=None):
    """Initialize the knapsack solver data structures by scaling the macs and the descending importance score.

    Args:
        layers_filters_latency (List[List[int]]): The latency cost list of the grouped neurons.
        layers_filters_macs (List[List[int]]): The latency cost list of the grouped neurons.
        importance_scores_filters (List[List[int]]): The importance score list in descending orders.
    Returns:
    """

    log_print("Constraints percentages: {}".format(constraints_percentages))
    macs_list = []
    layers_filters_byte_list = []

    for layer_index in range(len(descending_importance_score_scores)):

        for item_index in range(len(descending_importance_score_scores[layer_index])):
            descending_importance_score_scores[layer_index][item_index] = round(
                descending_importance_score_scores[layer_index][item_index] * int_scale_value)

    for layer_index in range(len(layers_filters_macs)):

        for item_index in range(len(layers_filters_macs[layer_index])):
            macs_list.append(layers_filters_macs[layer_index][item_index])
            layers_filters_byte_list.append(layers_filters_byte[layer_index][item_index])

            layers_filters_macs[layer_index][item_index] = round(
                layers_filters_macs[layer_index][item_index])  # / 10000

    macs_list_scaled = [round(filter_macs) for filter_macs in macs_list]  # / 10000

    initial_macs = sum(macs_list_scaled)
    initial_memory = sum(layers_filters_byte_list)
    macs_targets = []

    memory_targets = []

    # scale the initial target to be the subnetworks constraints
    for subnetworks_number in range(subnetworks_number - 1):
        macs_targets.append(round(initial_macs * constraints_percentages[subnetworks_number]))
        memory_targets.append(round(initial_memory * constraints_percentages[subnetworks_number]))

    log_print("Initial MACS: {} MACS targets: {}".format(initial_macs, macs_targets))

    return descending_importance_score_scores, layers_filters_macs, layers_filters_byte, macs_targets, memory_targets


def initialize_nested_knapsack_solver_ds_cnn(layers_filters_macs, layers_filters_byte,
                                             descending_importance_score_scores,
                                             int_scale_value=1e5, subnetworks_number=3, constraints_percentages=None):
    """Initialize the knapsack solver data structures by scaling the macs and the descending importance score.

    Args:
        layers_filters_latency (List[List[int]]): The latency cost list of the grouped neurons.
        layers_filters_macs (List[List[int]]): The latency cost list of the grouped neurons.
        importance_scores_filters (List[List[int]]): The importance score list in descending orders.
    Returns:
    """

    log_print("Constraints percentages: {}".format(constraints_percentages))
    macs_list = []
    layers_filters_byte_list = []

    for layer_index in range(len(descending_importance_score_scores)):

        for item_index in range(len(descending_importance_score_scores[layer_index])):
            descending_importance_score_scores[layer_index][item_index] = round(
                descending_importance_score_scores[layer_index][item_index] * int_scale_value)

    for layer_index in range(len(layers_filters_macs)):

        for item_index in range(len(layers_filters_macs[layer_index])):
            macs_list.append(layers_filters_macs[layer_index][item_index])
            layers_filters_byte_list.append(layers_filters_byte[layer_index][item_index])

            layers_filters_macs[layer_index][item_index] = round(layers_filters_macs[layer_index][item_index])

    # The smallest index of neurons belonging to each layer
    layer_index_split = []

    initial_index = 0
    for layer_index in range(len(layers_filters_macs)):

        if layer_index != 0:
            layer_index_split.append(len(layers_filters_macs[layer_index - 1]))
        initial_index += len(layers_filters_macs[layer_index])

    macs_list_scaled = [round(filter_macs) for filter_macs in macs_list]

    initial_macs = sum(macs_list_scaled)
    initial_memory = sum(layers_filters_byte_list)
    macs_targets = []

    memory_targets = []

    # scale the initial target to be the subnetworks constraints
    for subnetworks_number in range(subnetworks_number - 1):
        macs_targets.append(round(initial_macs * constraints_percentages[subnetworks_number]))
        memory_targets.append(round(initial_memory * constraints_percentages[subnetworks_number]))

    log_print("Initial MACS: {} MACS targets: {}".format(initial_macs, macs_targets))

    return descending_importance_score_scores, layers_filters_macs, layers_filters_byte, macs_targets, memory_targets


def initialize_nested_knapsack_solver_cnn(layers_filters_macs, layers_filters_byte,
                                          descending_importance_score_scores,
                                          int_scale_value=1e5, subnetworks_number=3, constraints_percentages=None):
    """Initialize the knapsack solver data structures by scaling the macs and the descending importance score.

    Args:
        layers_filters_latency (List[List[int]]): The latency cost list of the grouped neurons.
        layers_filters_macs (List[List[int]]): The latency cost list of the grouped neurons.
        importance_scores_filters (List[List[int]]): The importance score list in descending orders.
    Returns:
    """

    importance_list = []
    macs_list = []
    layers_filters_byte_list = []

    for layer_index in range(len(layers_filters_macs)):

        for item_index in range(len(layers_filters_macs[layer_index])):
            importance_list.append(descending_importance_score_scores[layer_index][item_index])
            macs_list.append(layers_filters_macs[layer_index][item_index])
            layers_filters_byte_list.append(layers_filters_byte[layer_index][item_index])

            descending_importance_score_scores[layer_index][item_index] = round(
                descending_importance_score_scores[layer_index][item_index] * int_scale_value)
            layers_filters_macs[layer_index][item_index] = round(layers_filters_macs[layer_index][item_index] )

    # The smallest index of neurons belonging to each layer
    layer_index_split = []

    initial_index = 0
    for layer_index in range(len(layers_filters_macs)):

        if layer_index != 0:
            layer_index_split.append(len(layers_filters_macs[layer_index - 1]))
        initial_index += len(layers_filters_macs[layer_index])

    macs_list_scaled = [round(filter_macs) for filter_macs in macs_list]

    initial_macs = sum(macs_list_scaled)
    initial_memory = sum(layers_filters_byte_list)
    macs_targets = []
    memory_targets = []

    # scale the initial target to be the subnetworks constraints
    for subnetworks_number in range(subnetworks_number - 1):
        macs_targets.append(round(initial_macs * constraints_percentages[subnetworks_number]))
        memory_targets.append(round(initial_memory * constraints_percentages[subnetworks_number]))

    return descending_importance_score_scores, layers_filters_macs, layers_filters_byte, macs_targets, memory_targets


def ortools_knapsack_solver_ds_cnn(layers, weights_macs_layers, weights_memory, filters_score, capacity_macs,
                                   capacity_memory_size, importance_score_pointwise_filters_kernels,
                                   units_layer_size=64,
                                   previous_optimal_solution_first_layer=None,
                                   previous_optimal_solution_feature_extraction=None,
                                   previous_optimal_solution_feature_creation=None, solver_time_limit=250,
                                   last_solution_iteration_search=False, last_pointwise_filters=1, model_size="s",
                                   solver_name="GUROBI"):
    """
    @param previous_optimal_solution: data structure used to store the previous optimal solution found
    @param layers: A list of lists (2D list) where each inner list represents a class (group) of items (number of units in a layer).
    @param weights_macs: A 2D list (list of lists) with the same structure as classes, but containing the weights of the items.
    @param filters_score: A 2D list (list of lists) with the same structure as classes, but containing the values of the items.
    @param capacity_macs: An integer or a float representing the maximum weight capacity of the knapsack.
    @return: a List[int] indicating the number of filters to use for each Convolution Layer
    """
    solver = pywraplp.Solver.CreateSolver(solver_name)
    solver.EnableOutput()
    solver.SetSolverSpecificParametersAsString('LogToConsole=1')

    print("Solver name: {} Running time: {}ms".format(solver_name, int(solver_time_limit * 1000)))

    solver.SetTimeLimit(int(solver_time_limit * 1000))

    # all the 1x1 kernels have the same weight in terms of MACs
    mac_individual_kernel_pointwise_layers = [math.ceil(weights_macs_layers[i][0] / len(weights_macs_layers[i - 1])) for
                                              i in
                                              range(2, len(weights_macs_layers), 2)]

    weights_macs = []
    if model_size == "l":
        for depthwise_layer_index in [0, 1, 3, 5, 7, 9]:
            weights_macs.append(weights_macs_layers[depthwise_layer_index])
    elif model_size == "m" or model_size == "s":
        for depthwise_layer_index in [0, 1, 3, 5, 7]:
            weights_macs.append(weights_macs_layers[depthwise_layer_index])

    depthwise_filters = []
    first_convolution_layer = [solver.BoolVar(f'y_-1_{unit_index}') for unit_index in range(len(layers[0]))]

    for layer_index, layer_units in enumerate(layers[1:]):
        depthwise_filters.append(
            [solver.BoolVar(f'dw_{layer_index}_{unit_index}') for unit_index in range(len(layer_units))])

    # Objective function: sum of the importance score of the depthwise filters
    objective = solver.Objective()
    for unit_index in range(len(first_convolution_layer)):
        objective.SetCoefficient(first_convolution_layer[unit_index], filters_score[0][unit_index])

    for layer_index, layer_units in enumerate(depthwise_filters):
        for unit_index in range(len(layer_units)):
            objective.SetCoefficient(depthwise_filters[layer_index][unit_index],
                                     filters_score[layer_index + 1][unit_index])

    # Boolean Var: create one boolean variable for each pointwise filter of each pointwise layer
    pointwise_filters = []
    for layer_index, layer_units in enumerate(layers[1:]):
        pointwise_filters.append(
            [solver.BoolVar(f'pf_k_{layer_index}_{unit_index}') for unit_index in range(len(layer_units))])

    # Integer Var: create one integer variable for each pointwise layer and for the first convolution layer
    x_0 = solver.IntVar(0, units_layer_size, 'x_-1')
    x_i = [solver.IntVar(0, units_layer_size, f'x_{unit_index}') for unit_index in range(len(pointwise_filters))]

    # Initialize the integer variables
    solver.Add(x_0 == solver.Sum(first_convolution_layer))
    for pointwise_filters_index in range(len(pointwise_filters)):
        solver.Add(x_i[pointwise_filters_index] == solver.Sum(pointwise_filters[pointwise_filters_index]))

    # Boolean Var: create one boolean variable for each kernel of each pointwise filter of each pointwise layer
    pointwise_filters_kernels = [[] for _ in range(len(pointwise_filters))]
    for pointwise_layer_index in range(len(pointwise_filters)):

        for pointwise_filter_index in range(len(pointwise_filters[pointwise_layer_index])):
            pointwise_filters_kernels[pointwise_layer_index].append(
                [solver.BoolVar(f'pf_kt_{pointwise_layer_index}_{pointwise_filter_index}_{pointwise_kernel_index}')
                 for pointwise_kernel_index in range(units_layer_size)])

    # Add Pointwise filters kernels importance score to the objective function
    for pointwise_layer_index in range(len(pointwise_filters)):
        for pointwise_filter_index in range(len(pointwise_filters[pointwise_layer_index])):
            for pointwise_kernel_index in range(units_layer_size):
                objective.SetCoefficient(
                    pointwise_filters_kernels[pointwise_layer_index][pointwise_filter_index][pointwise_kernel_index],
                    importance_score_pointwise_filters_kernels[pointwise_layer_index][pointwise_filter_index][
                        pointwise_kernel_index])

    objective.SetMaximization()

    # Define MACs constraint (capacity constraint)
    capacity_constraint = solver.Constraint(0, capacity_macs, "ct")

    # Constraint: set to one the solutions previously found for the feature extraction and creation filters
    if previous_optimal_solution_first_layer is not None:

        for unit_index in range(len(first_convolution_layer)):
            if previous_optimal_solution_first_layer[unit_index] == 1:
                solver.Add(first_convolution_layer[unit_index] == 1)

    if previous_optimal_solution_feature_creation is not None:

        for layer_index, layer_units in enumerate(pointwise_filters):
            for unit_index in range(len(layer_units)):
                if previous_optimal_solution_feature_creation[layer_index][unit_index] == 1:
                    solver.Add(pointwise_filters[layer_index][unit_index] == 1)

    if previous_optimal_solution_feature_extraction is not None:

        for layer_index, layer_units in enumerate(depthwise_filters):
            for unit_index in range(len(layer_units)):
                if previous_optimal_solution_feature_extraction[layer_index][unit_index] == 1:
                    solver.Add(depthwise_filters[layer_index][unit_index] == 1)

    # Constraint: first filter in standard convolution layer must be equal to number of filters in the first depthwise layer
    solver.Add(x_0 == solver.Sum(depthwise_filters[0]))
    for depthwise_layer_index in range(1, len(depthwise_filters)):
        solver.Add(x_i[depthwise_layer_index - 1] == solver.Sum(depthwise_filters[depthwise_layer_index]))

    # Constraint: add first layer filters weights to the capacity constraint (first layer filters are standard convolution filters)
    for standard_convolution_filter_index in range(len(first_convolution_layer)):
        capacity_constraint.SetCoefficient(first_convolution_layer[standard_convolution_filter_index],
                                           weights_macs[0][standard_convolution_filter_index])

    # Constraint: impose the pointwise filters to be taken in ascending order to preserve the memory layout
    for layer_index, layer_units in enumerate(pointwise_filters):
        for unit_index in range(len(layer_units) - 1):
            solver.Add(pointwise_filters[layer_index][unit_index] >= pointwise_filters[layer_index][unit_index + 1])

    # Constraint: for each layer take at least one filter and the last number of pointwise filters to be >= than last_pointwise_filters
    solver.Add(x_0 >= 1)
    for layer_index, layer_units in enumerate(pointwise_filters):
        if layer_index == len(pointwise_filters) - 1:
            solver.Add(x_i[layer_index] >= last_pointwise_filters)
        else:
            solver.Add(x_i[layer_index] >= 1)

    # Constraint: add depthwise layer filters weights to the capacity constraint
    weights_macs_depthwise = weights_macs[1:]
    for layer_index, layer_units in enumerate(depthwise_filters):

        for unit_index in range(len(layer_units)):

            capacity_constraint.SetCoefficient(depthwise_filters[layer_index][unit_index],
                                               weights_macs_depthwise[layer_index][
                                                   unit_index])

            for pointwise_kernel_index in range(len(pointwise_filters_kernels[layer_index][unit_index])):
                capacity_constraint.SetCoefficient(
                    pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index],
                    int(mac_individual_kernel_pointwise_layers[layer_index]))

    # Constraint: if taken a kernel t of filter k then I have to take the filter k of the pointwise layers
    for layer_index, layer_units in enumerate(depthwise_filters):
        for unit_index in range(len(layer_units)):
            for pointwise_kernel_index in range(len(pointwise_filters_kernels[layer_index][unit_index])):
                solver.Add(pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index] <=
                           pointwise_filters[layer_index][unit_index])

    # Constraint (2):
    for layer_index, layer_units in enumerate(pointwise_filters):
        for unit_index in range(len(layer_units)):
            solver.Add(pointwise_filters[layer_index][unit_index] <=
                       solver.Sum(pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index] for
                                  pointwise_kernel_index in
                                  range(len(pointwise_filters_kernels[layer_index][unit_index]))))

    # Constraint: the number of kernel in the filter k of pointwise layer i must be less or equal than the number of filters taken in the layer before
    for layer_index, layer_units in enumerate(pointwise_filters_kernels):

        for unit_index in range(len(layer_units)):
            if layer_index == 0:
                solver.Add(solver.Sum(
                    pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index] for
                    pointwise_kernel_index in range(len(layer_units))) <=
                           x_0)

            else:
                solver.Add(solver.Sum(
                    pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index] for
                    pointwise_kernel_index in range(len(layer_units))) <=
                           x_i[layer_index - 1])

    # Constraint: if you take the filter k at layer i t the number of kernels t of filter k in layer i has
    # to be equal than the numer of pointwise filters in the block before
    # (number of filters of depthwise layer i) if you do not take the filter
    # k at layer i, than constraint (3) and (4) together imply that all the
    # kernels t of filter k at layer i have to be zero
    for layer_index, layer_units in enumerate(pointwise_filters_kernels):

        for unit_index in range(len(layer_units)):

            if layer_index == 0:

                solver.Add(solver.Sum(
                    pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index] for
                    pointwise_kernel_index in range(len(layer_units))) >=
                           x_0 - (1 - pointwise_filters[layer_index][unit_index]) * units_layer_size)

            else:

                solver.Add(solver.Sum(
                    pointwise_filters_kernels[layer_index][unit_index][pointwise_kernel_index] for
                    pointwise_kernel_index in range(len(layer_units))) >=
                           x_i[layer_index - 1] - (1 - pointwise_filters[layer_index][unit_index]) * units_layer_size)

    # Print model to file for debug purposes
    # log_print(solver.ExportModelAsLpFormat(False), printing=False)
    # Solve
    start_time = time.time()
    status = solver.Solve()
    end_time = time.time()
    elapsed_time = end_time - start_time
    log_print(f"Elapsed time: {elapsed_time} seconds")

    first_convolution_layer_indexes = []  # indexes of the first convolution layer
    depthwise_indexes = [[] for _ in range(len(depthwise_filters))]
    pointwise_indexes = [[] for _ in range(len(pointwise_filters))]

    # iterate over the bool var to save the solution found by the knapsack solver
    for first_convolution_filter_index in range(len(first_convolution_layer)):
        if first_convolution_layer[first_convolution_filter_index].solution_value() == 1:
            first_convolution_layer_indexes.append(first_convolution_filter_index)

    for depthwise_layer_index in range(len(depthwise_filters)):

        for unit_index in range(len(depthwise_filters[depthwise_layer_index])):

            if depthwise_filters[depthwise_layer_index][unit_index].solution_value() == 1:
                depthwise_indexes[depthwise_layer_index].append(unit_index)

    for pointwise_layer_index in range(len(pointwise_filters)):

        for unit_index in range(len(pointwise_filters[pointwise_layer_index])):

            if pointwise_filters[pointwise_layer_index][unit_index].solution_value() == 1:
                pointwise_indexes[pointwise_layer_index].append(unit_index)

    solution_macs = 0  # solution macs of the solution found by the knapsack solver

    # add the macs of the first convolution layer
    solution_macs += sum([weights_macs[0][index] for index in first_convolution_layer_indexes])
    print("First convolution layer MACs: {}".format(solution_macs))

    depthwise_macs = weights_macs[1:]
    for block_index in range(len(depthwise_indexes)):
        block_macs = 0
        block_macs += sum([depthwise_macs[block_index][index] for index in depthwise_indexes[block_index]])
        block_macs += len(pointwise_indexes[block_index]) * len(depthwise_indexes[block_index])
        print("Current block MACs: " + str(block_macs))
        solution_macs += block_macs

    print("MACs capacity: " + str(capacity_macs), "Solution MACs: " + str(solution_macs))

    # A solution has been found (or we are in the last iteration), save the solutions found
    if solution_macs <= capacity_macs or last_solution_iteration_search:

        # iterate over the bool var to save the solution found by the knapsack solver
        for first_convolution_filter_index in range(len(first_convolution_layer)):

            if first_convolution_layer[first_convolution_filter_index].solution_value() == 1:
                if previous_optimal_solution_first_layer is not None and \
                        previous_optimal_solution_first_layer[first_convolution_filter_index] == 0:
                    previous_optimal_solution_first_layer[first_convolution_filter_index] = 1

        for depthwise_layer_index in range(len(depthwise_filters)):

            for unit_index in range(len(depthwise_filters[depthwise_layer_index])):
                if depthwise_filters[depthwise_layer_index][unit_index].solution_value() == 1:
                    if previous_optimal_solution_feature_extraction is not None and \
                            previous_optimal_solution_feature_extraction[depthwise_layer_index][unit_index] == 0:
                        previous_optimal_solution_feature_extraction[depthwise_layer_index][unit_index] = 1

        for pointwise_layer_index in range(len(pointwise_filters)):

            for unit_index in range(len(pointwise_filters[pointwise_layer_index])):
                if pointwise_filters[pointwise_layer_index][unit_index].solution_value() == 1:
                    if previous_optimal_solution_feature_creation is not None and \
                            previous_optimal_solution_feature_creation[pointwise_layer_index][unit_index] == 0:
                        previous_optimal_solution_feature_creation[pointwise_layer_index][unit_index] = 1

    log_print("First convolution layer indexes: ")
    log_print(first_convolution_layer_indexes)
    log_print("Depthwise layer indexes: ")
    log_print(depthwise_indexes)
    log_print("Pointwise layer indexes: ")
    log_print(pointwise_indexes)
    return max(first_convolution_layer_indexes), [max(filter_indexes) for
                                                  filter_indexes in
                                                  depthwise_indexes], [
        max(filter_indexes) for filter_indexes in pointwise_indexes], solution_macs


def ortools_knapsack_solver_dnn(classes, weights_macs, weights_memory, filters_score, capacity_macs,
                                capacity_memory_size,
                                previous_optimal_solution=None, bottom_up=True, solver_name='GUROBI'):
    """

    @param previous_optimal_solution: data structure used to store the previous optimal solution found
    @param classes: A list of lists (2D list) where each inner list represents a class (group) of items (number of units in a layer).
    @param weights_macs: A 2D list (list of lists) with the same structure as classes, but containing the weights of the items.
    @param filters_score: A 2D list (list of lists) with the same structure as classes, but containing the values of the items.
    @param capacity_macs: An integer or a float representing the maximum weight capacity of the knapsack.
    @return: a List[int] indicating the number of filters to use for each Convolution Layer
    """
    solver = pywraplp.Solver.CreateSolver(solver_name)
    #solver.EnableOutput()

    # Create binary variables for each computational unit
    x = []
    for i, items in enumerate(classes):
        x.append([solver.BoolVar(f'x_{i}_{j}') for j in range(len(items))])

    # Objective function: maximize total value, cast the values to integer
    solver.Maximize(
        solver.Sum(x[i][j] * int(filters_score[i][j]) for i in range(len(classes)) for j in range(len(classes[i]))))

    # Constraint: limit the total number of MACs
    solver.Add(
        solver.Sum(
            x[i][j] * int(weights_macs[i][j]) for i in range(len(classes)) for j in range(len(classes[i]))) <= int(
            capacity_macs))

    # Constraint: taking on variable for each layer at least
    x_i = [solver.IntVar(0, len(classes[i]), f'x_{i}') for i in range(0, len(classes))]
    for i in range(0, len(classes)):
        solver.Add(x_i[i] == solver.Sum(x[i]))
        solver.Add(x_i[i] >= 2)

    if bottom_up:
        # Constraint: select the items previously found by the knapsack solver
        for i in range(len(classes)):
            for j in range(len(classes[i])):
                if previous_optimal_solution[i][j] == 1:
                    solver.Add(x[i][j] == 1)

    # Solve
    status = solver.Solve()
    filters_indexes = [[] for _ in range(len(classes))]

    if status == pywraplp.Solver.OPTIMAL:

        for i in range(len(classes)):
            for j in range(len(classes[i])):
                if x[i][j].solution_value() == 1:
                    filters_indexes[i].append(j)

                    if bottom_up:
                        if previous_optimal_solution[i][j] == 0:
                            previous_optimal_solution[i][j] = 1
    else:
        print('The problem does not have an optimal solution.')

    solution_macs = 0
    solution_macs += sum([weights_macs[0][index] for index in filters_indexes[0]])
    solution_macs += sum([weights_macs[1][index] for index in filters_indexes[1]])
    solution_macs += sum([weights_macs[2][index] for index in filters_indexes[2]])
    print("Subnetworks MACs: {}".format(solution_macs))


    return [max(filter_indexes) for filter_indexes in filters_indexes]


def ortools_knapsack_solver_cnn(classes, weights_macs, weights_memory, capacity_memory_size, filters_score,
                                capacity_macs,
                                previous_optimal_solution=None, bottom_up=True, solver_name='GUROBI'):
    """

    @param previous_optimal_solution: data structure used to store the previous optimal solution found
    @param classes: A list of lists (2D list) where each inner list represents a class (group) of items (number of units in a layer).
    @param weights_macs: A 2D list (list of lists) with the same structure as classes, but containing the weights of the items.
    @param filters_score: A 2D list (list of lists) with the same structure as classes, but containing the values of the items.
    @param capacity_macs: An integer or a float representing the maximum weight capacity of the knapsack.
    @return: a List[int] indicating the number of filters to use for each Convolution Layer
    """
    solver = pywraplp.Solver.CreateSolver(solver_name)
    #solver.EnableOutput()

    # Create binary variables for each computational unit
    x = []
    for i, items in enumerate(classes):
        x.append([solver.BoolVar(f'x_{i}_{j}') for j in range(len(items))])

    # Objective function: maximize total value, cast the values to integer
    solver.Maximize(
        solver.Sum(x[i][j] * int(filters_score[i][j]) for i in range(len(classes)) for j in range(len(classes[i]))))

    # Constraint: limit the total number of MACs
    solver.Add(
        solver.Sum(
            x[i][j] * int(weights_macs[i][j]) for i in range(len(classes)) for j in range(len(classes[i]))) <= int(
            capacity_macs))

    x_i = [solver.IntVar(0, len(classes[i]), f'x_{i}') for i in range(0, len(classes))]
    for i in range(0, len(classes)):
        solver.Add(x_i[i] == solver.Sum(x[i]))
        solver.Add(x_i[i] >= 2)

    if bottom_up:
        # Constraint: select the items previously found by the knapsack solver
        for i in range(len(classes)):
            for j in range(len(classes[i])):
                if previous_optimal_solution[i][j] == 1:
                    solver.Add(x[i][j] == 1)

    # Solve
    status = solver.Solve()
    filters_indexes = [[] for _ in range(len(classes))]

    if status == pywraplp.Solver.OPTIMAL:

        for i in range(len(classes)):
            for j in range(len(classes[i])):
                if x[i][j].solution_value() == 1:
                    filters_indexes[i].append(j)

                    if bottom_up:
                        # save new solution found by the knapsack solver for the next iteration
                        if previous_optimal_solution[i][j] == 0:
                            previous_optimal_solution[i][j] = 1

        print("Solution MACS: {}".format(sum([weights_macs[i][index] for i, index in enumerate(filters_indexes)])))
    else:
        print('The problem does not have an optimal solution.')

    return [max(filter_indexes) for filter_indexes in filters_indexes]
