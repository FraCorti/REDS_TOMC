"""
Vision Transform Integer Linear Programming Structural Pruning formulation
===================================

This script provides the integer linear programming structural pruning formulation for Vision Transformer base model.

Author: Francesco Corti (francesco.corti@tugraz.at)
Date: 2025-07-29

"""
import argparse
import os

import timm
from datasets import load_dataset
import torch
from ortools.linear_solver import pywraplp
from timm.data import resolve_data_config, create_transform


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(model_name=args.model_name,
                              pretrained=True)
    model.eval()
    model.to(device=device)

    imagenet_training_set = load_dataset(path='timm/imagenet-1k-wds', split="train", streaming=True,
                                         token=args.hugginface_token,
                                         cache_dir="{}/datasets/timm/imagenet-1k-wds/train/".format(
                                             os.getcwd()))
    config = resolve_data_config({}, model=model)
    imagenet_preprocessing_training_set = create_transform(**config, is_training=True)

    def _collate_fn_imagenet_training_set(examples):
        images = []
        labels = []

        for example in examples:
            images.append((imagenet_preprocessing_training_set(example["jpg"].convert("RGB"))))
            labels.append(example["json"]["label"])

        images = torch.stack(images)
        labels = torch.tensor(labels)

        return images, labels

    def compute_score_cost(model, trainloader, device, batches=1000):
        model.zero_grad()

        # accumulating gradient scores for all the batches
        for i in range(batches):

            inputs, targets = next(iter(trainloader))
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass and gradient accumulation
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()

        # Dimensions and gradient-based scores/costs
        conv_grad = model.patch_embed.proj.weight.grad  # [E, C, P, P]
        total_convolution_filters_first_layer = conv_grad.size(0)
        head_number = total_convolution_filters_first_layer // 64

        # Score and cost for patch filters (conv + QKV)
        score_head_convolutional_filters = (conv_grad * model.patch_embed.proj.weight).sum(
            dim=(1, 2, 3)).abs().tolist()
        # score_head_convolutional_filters = [0.0 for _ in range(total_convolution_filters_first_layer)]
        score_mlp1_blocks = []
        score_mlp2_blocks = []

        # compute score for all blocks
        for block in model.blocks:

            # [3* len(score_head_convolutional_filters), score_head_convolutional_filters]
            qkv_score = (block.attn.qkv.weight.grad * block.attn.qkv.weight)

            for conv_filter_index in range(total_convolution_filters_first_layer):
                # Query matrix head score
                score_head_convolutional_filters[conv_filter_index] += qkv_score[:conv_filter_index + 1,
                                                                       conv_filter_index].sum().abs().item()
                score_head_convolutional_filters[conv_filter_index] += qkv_score[conv_filter_index,
                                                                       :conv_filter_index + 1].sum().abs().item()
                score_head_convolutional_filters[conv_filter_index] -= qkv_score[conv_filter_index,
                conv_filter_index].abs().item()

                # Key matrix head score
                score_head_convolutional_filters[conv_filter_index] += (qkv_score[
                                                                        len(score_head_convolutional_filters): len(
                                                                            score_head_convolutional_filters) * 2,
                                                                        0:len(score_head_convolutional_filters)])[
                                                                       :conv_filter_index + 1,
                                                                       conv_filter_index].sum().abs().item()
                score_head_convolutional_filters[conv_filter_index] += (qkv_score[
                                                                        len(score_head_convolutional_filters): len(
                                                                            score_head_convolutional_filters) * 2,
                                                                        0:len(score_head_convolutional_filters)])[
                                                                       conv_filter_index,
                                                                       :conv_filter_index + 1].sum().abs().item()
                score_head_convolutional_filters[conv_filter_index] -= (qkv_score[
                                                                        len(score_head_convolutional_filters): len(
                                                                            score_head_convolutional_filters) * 2,
                                                                        0:len(score_head_convolutional_filters)])[
                    conv_filter_index,
                    conv_filter_index].abs().item()

                # Value matrix head score
                score_head_convolutional_filters[conv_filter_index] += (qkv_score[
                                                                        len(score_head_convolutional_filters): len(
                                                                            score_head_convolutional_filters) * 3,
                                                                        0:len(score_head_convolutional_filters)])[
                                                                       :conv_filter_index + 1,
                                                                       conv_filter_index].sum().abs().item()
                score_head_convolutional_filters[conv_filter_index] += (qkv_score[
                                                                        len(score_head_convolutional_filters): len(
                                                                            score_head_convolutional_filters) * 3,
                                                                        0:len(score_head_convolutional_filters)])[
                                                                       conv_filter_index,
                                                                       :conv_filter_index + 1].sum().abs().item()
                score_head_convolutional_filters[conv_filter_index] -= (qkv_score[
                                                                        len(score_head_convolutional_filters): len(
                                                                            score_head_convolutional_filters) * 3,
                                                                        0:len(score_head_convolutional_filters)])[
                    conv_filter_index,
                    conv_filter_index].abs().item()

            proj_score = (block.attn.proj.weight.grad * block.attn.proj.weight)

            for conv_filter_index in range(total_convolution_filters_first_layer):
                proj_score_row = proj_score[conv_filter_index, :conv_filter_index + 1].sum().abs().item()
                proj_score_column = proj_score[:conv_filter_index + 1, conv_filter_index].sum().abs().item()
                proj_score_head = proj_score_row + proj_score_column - proj_score[
                    conv_filter_index, conv_filter_index].abs().item()

                score_head_convolutional_filters[conv_filter_index] += proj_score_head

            # MLP1
            ml1_grad = block.mlp.fc1.weight.grad  # [len(score_head_convolutional_filters), M]
            ml1_weight = block.mlp.fc1.weight
            ml1_score = torch.zeros(size=(int(block.mlp.fc1.weight.size(0) / 64), head_number), device=device)
            for neuron_mlp1 in range(int(block.mlp.fc1.weight.size(0) / 64)):
                for head in range(head_number):
                    ml1_score[neuron_mlp1][head] = (ml1_weight[neuron_mlp1, head * 64: (head + 1) * 64] * ml1_grad[
                                                                                                          neuron_mlp1,
                                                                                                          head * 64: (
                                                                                                                             head + 1) * 64]).sum().abs().item()

            score_mlp1_blocks.append(ml1_score)

            # MLP2
            ml2_grad = block.mlp.fc2.weight.grad  # [M, len(score_head_convolutional_filters)]
            ml2_weight = block.mlp.fc2.weight
            ml2_score = torch.zeros(size=(head_number, int(block.mlp.fc2.weight.size(1) / 64)), device=device)
            for head in range(head_number):
                for neuron_mlp2 in range(int(block.mlp.fc2.weight.size(1) / 64)):
                    ml2_score[head][neuron_mlp2] = (ml2_weight[neuron_mlp2, head * 64: (head + 1) * 64] * ml2_grad[
                                                                                                          neuron_mlp2,
                                                                                                          head * 64: (
                                                                                                                             head + 1) * 64]).sum().abs().item()

            score_mlp2_blocks.append(ml2_score)

        model.eval()
        
        # MACs obtained from the DeepSpeed profiler: https://www.deepspeed.ai/tutorials/flops-profiler/ 
        macs_head = 115605504
        macs_qkc = 348585984
        macs_proj = 175869952
        macs_mlp1 = 464781312
        macs_mlp2 = 464781312
        cost_heads = []

        sample_matrix = torch.zeros(size=(head_number, head_number))
        # Cost for every head taken which implies a cost for qkv and projection matrices in all the blocks
        for h in range(head_number):
            macs_single_head_initial_convolutional_filters = macs_head / 12
            macs_single_head_qkv = (macs_qkc / 432) * (
                    len(sample_matrix[h, :h + 1]) + len(sample_matrix[:h + 1, h])) * 3  # for QKV

            macs_single_head_projection = (macs_proj / 144) * (
                    len(sample_matrix[h, :h + 1]) + len(sample_matrix[:h + 1, h]))  # 1 for projection

            macs_all_model_qkv_projection = macs_single_head_qkv * len(
                model.blocks) + macs_single_head_projection * len(model.blocks)
            cost_heads.append(int(macs_all_model_qkv_projection + macs_single_head_initial_convolutional_filters))

        cost_single_mlp1 = torch.zeros(size=(int(model.blocks[0].mlp.fc1.weight.size(0) / 64), head_number),
                                       device=device, dtype=torch.int32)
        cost_single_mlp2 = torch.zeros(size=(head_number, int(model.blocks[0].mlp.fc2.weight.size(1) / 64)),
                                       device=device, dtype=torch.int32)

        for h in range(head_number):
            # Cost for MLP1
            for k in range(int(model.blocks[0].mlp.fc1.weight.size(0) / 64)):
                cost_single_mlp1[k][h] += int(
                    (macs_mlp1 / head_number) / int(model.blocks[0].mlp.fc1.weight.size(0) / 64))
            # Cost for MLP2
            for k in range(int(model.blocks[0].mlp.fc2.weight.size(1) / 64)):
                cost_single_mlp2[h][k] += int(
                    (macs_mlp2 / head_number) / int(model.blocks[0].mlp.fc2.weight.size(1) / 64))

        score_head = []
        for h in range(head_number):
            score_head.append(sum(score_head_convolutional_filters[h * 64: (h + 1) * 64]))

        score = {
            'head': score_head,
            'mlp1': score_mlp1_blocks,
            'mlp2': score_mlp2_blocks
        }
        cost = {
            'head': cost_heads,
            'mlp1': cost_single_mlp1,
            'mlp2': cost_single_mlp2
        }

        macs_solver = sum(cost_heads) + cost_single_mlp1.sum().item() * len(
            model.blocks) + cost_single_mlp2.sum().item() * len(
            model.blocks)
        return score, cost, macs_solver

    # Build ILP: select heads, MLP1 and MLP2 neurons per block
    # Include explicit integer variables for head count (H_count) and embedding dim (D_prime)
    def build_pruning_ilp(score=None, cost=None, budget=None, blocks_number=None):
        solver = pywraplp.Solver.CreateSolver('GUROBI')  # SCIP
        if not solver:
            raise RuntimeError('Gurobi solver unavailable; ensure OR-Tools built with Gurobi')

        head_number = len(score['head'])
        neuron_mlp1 = len(score['mlp1'][0])  # 48
        input_features_mlp1 = len(score['mlp1'])  # 12

        neuron_mlp2 = len(score['mlp2'][0])  # 12
        input_features_mlp2 = len(score['mlp2'][0][1])  # 48

        # Create binary decision variables
        y = [solver.BoolVar(f'y_[{i}]') for i in range(head_number)]

        f = []
        for t in range(blocks_number):
            layer_mlp1 = []
            for k in range(neuron_mlp1):
                neuron_binary_variables_mlp1 = []
                for i in range(input_features_mlp1):
                    neuron_binary_variables_mlp1.append(solver.BoolVar(f'f_{k}_{i}^{t}'))
                layer_mlp1.append(neuron_binary_variables_mlp1)
            f.append(layer_mlp1)

        g = []
        for t in range(blocks_number):
            layer_mlp2 = []
            for k in range(neuron_mlp2):
                neuron_binary_variables_mlp2 = []
                for i in range(input_features_mlp2):
                    neuron_binary_variables_mlp2.append(solver.BoolVar(f'g_{k}_{i}^{t}'))
                layer_mlp2.append(neuron_binary_variables_mlp2)
            g.append(layer_mlp2)

        # Integer vars definition and initialization
        x_0 = solver.IntVar(0, len(y), 'x_0')
        n_mlp1 = [solver.IntVar(0, len(f[t]), f'n_mlp1_{t}') for t in range(blocks_number)]

        solver.Add(x_0 == solver.Sum(y))
        for t in range(blocks_number):
            solver.Add(n_mlp1[t] == solver.Sum(f[t][i][0] for i in range(len(f[t]))))

        # Objective
        obj = solver.Objective()
        obj.SetMaximization()
        capacity_constraint = solver.Constraint(0, budget, "ct")
        for i in range(head_number):
            obj.SetCoefficient(y[i], score['head'][i])
            capacity_constraint.SetCoefficient(y[i], cost['head'][i])

        for t in range(blocks_number):

            for i in range(len(f[t])):
                for k in range(input_features_mlp1):
                    obj.SetCoefficient(f[t][i][k], score['mlp1'][t][i][k].item())
                    capacity_constraint.SetCoefficient(f[t][i][k], cost['mlp1'][i, k].item())

            for i in range(len(g[t])):
                for k in range(input_features_mlp2):
                    obj.SetCoefficient(g[t][i][k], score['mlp2'][t][i][k].item())
                    capacity_constraint.SetCoefficient(g[t][i][k], cost['mlp2'][i, k].item())

        # Constraint (13): the number of input feature for MLP1^t is equal to the number of head chosen x_0.
        for t in range(blocks_number):
            solver.Add(solver.Sum(f[t][0][k] for k in range(len(f[t][0]))) == x_0)

        # Constraint (14): The number of columns (i.e. neuron's input weight)
        # of each first row MLP_2^t is equal to the number of neurons chosen in MLP1^t.
        for t in range(blocks_number):
            solver.Add(
                solver.Sum(g[t][0][i] for i in range(len(g[t][0]))) == n_mlp1[
                    t])

        # Constraint (15): If a neuron f_i is picked, then its number of input weights connections
        # is equal to the number of heads convolutional filters chosen x_0.
        for t in range(blocks_number):
            for k in range(len(f[t][0])):
                solver.Add(solver.Sum(f[t][i][k] for i in range(len(y))) >= x_0 - len(y) * (1 - f[t][0][k]))

        # Constraint (16): The number of neurons chosen in MLP2^t is equal to the number of
        # heads convolutional filters chosen x_0 and the sum of each neuron input weights is equal
        # to the number of neurons chosen in the previous mlp_1^t layer.
        for t in range(blocks_number):
            for k in range(len(g[t][0])):
                solver.Add(
                    solver.Sum(g[t][i][k] for i in range(len(y))) >= x_0 - len(y) * (1 - g[t][0][k]))

        # Constraint (17): the number of convolutional filters heads should be taken in ascending order .
        for head_number in range(len(y) - 1):
            solver.Add(y[head_number] >= y[head_number + 1])

        # Constraint (18): If head convolutional filters y_i is not chosen then all
        # the corresponding columns in mlp1_t and rows in mlp2_t are equal to zero.
        for t in range(blocks_number):
            for i in range(len(f[t])):
                for k in range(len(f[t][i])):
                    solver.Add(f[t][i][k] <= y[k])

            for i in range(len(g[t])):
                for k in range(len(g[t][i])):
                    solver.Add(g[t][i][k] <= y[i])

        # Constraint (19): the number of input weight connections in MLP1 and MLP2 neurons should be taken in ascending order.
        for t in range(blocks_number):
            for i in range(len(f[t])):
                for k in range(len(f[t][i]) - 1):
                    solver.Add(f[t][i][k] >=
                               f[t][i][k + 1])

            for i in range(len(g[t])):
                for k in range(len(g[t][i]) - 1):
                    solver.Add(g[t][i][k] >=
                               g[t][i][k + 1])

        # Constraint (20): the number of MLP1 and MLP2 neurons should be taken in ascending order.
        for t in range(blocks_number):
            for i in range(len(f[t]) - 1):
                solver.Add(f[t][i][0] >= f[t][i + 1][0])

            for i in range(len(g[t]) - 1):
                solver.Add(g[t][i][0] >= g[t][i + 1][0])

        # Constraint (21): the number of MLP1 and MLP2 neurons should be mayor than 1.
        for t in range(blocks_number):
            solver.Add(solver.Sum(f[t][i][0] for i in range(len(f[t]))) >= 1)
            solver.Add(solver.Sum(g[t][i][0] for i in range(len(g[t]))) >= 1)

        return solver, y, f, g, capacity_constraint

    vit_heads = {0: "3 heads", 1: "6 heads"}
    index = 0
    for mac_budgets in [1253683200 - 192000, 4598882304 - 384000]:

        trainloader = torch.utils.data.DataLoader(
            imagenet_training_set,
            batch_size=256,
            collate_fn=_collate_fn_imagenet_training_set
        )

        score, cost, budget = compute_score_cost(model=model, trainloader=trainloader, device=device)

        # Set your FLOPs budget
        solver, y, binary_variables_mlp1, binary_variables_mlp2, capacity_constraint = build_pruning_ilp(score=score,
                                                                                                         cost=cost,
                                                                                                         budget=mac_budgets,
                                                                                                         blocks_number=len(
                                                                                                             model.blocks))
        # Compute total cost for head, mlp1 and mlp2

        print("Constraint", capacity_constraint.name(), "bounds = [", capacity_constraint.Lb(), ",",
              capacity_constraint.Ub(), "]")
        solver.EnableOutput()
        solver.SetSolverSpecificParametersAsString('LogToConsole=1')
        status = solver.Solve()

        total_macs = sum(capacity_constraint.GetCoefficient(v) * v.solution_value()
                         for v in solver.variables()
                         if capacity_constraint.GetCoefficient(v) != 0)
        subnetwork_solutions = [[] for _ in range(len(model.blocks) + 1)]

        if status == pywraplp.Solver.OPTIMAL:
            heads = [h for h, v in enumerate(y) if v.solution_value() == 1]

            # Extract selected neurons for MLP1 and MLP2 from the binary variables for all block t
            mlp1 = []
            mlp2 = []
            for t in range(len(model.blocks)):
                mlp1_neuron_layer = []
                for i in range(len(binary_variables_mlp1[t])):
                    if binary_variables_mlp1[t][i][0].solution_value() == 1:
                        mlp1_neuron_layer.append(i)
                mlp1.append(mlp1_neuron_layer)

                mlp2_neuron_layer = []
                for i in range(len(binary_variables_mlp2[t])):
                    if binary_variables_mlp2[t][i][0].solution_value() == 1:
                        mlp2_neuron_layer.append(i)
                mlp2.append(mlp2_neuron_layer)

            # Print the selected heads and neurons
            print("Attention Heads:", heads)
            for block_index in range(len(model.blocks)):
                print(f"MLP1 neurons in block {block_index}:", mlp1[block_index])
                print(f"MLP2 neurons in block {block_index}:", mlp2[block_index])

            # Append final solutions to the hydravit list
            subnetwork_solutions[0].append(heads)
            for block_index in range(len(model.blocks)):
                subnetwork_solutions[block_index + 1].append(mlp1[block_index])
                subnetwork_solutions[block_index + 1].append(mlp2[block_index])

        else:
            print("No optimal solution found.")

        # Print the final solutions
        print("Final solutions for {} budget, equal to heads: {} total MACs solution: {}".format(
            mac_budgets,
            vit_heads[index],
            total_macs))

        print(subnetwork_solutions)
        index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Integer Linear Programming Pruning for Vision Transformer')
    parser.add_argument('--gurobi_home', type=str, default='add GUROBI_HOME here',
                        help='Path to the Gurobi installation directory')
    parser.add_argument('--gurobi_license_file', type=str, default='add GUROBI_LICENSE_FILE here',
                        help='Path to the Gurobi license file')
    parser.add_argument('--cuda_device', type=int, default=-1)
    parser.add_argument('--model_name', type=str, default='deit_base_patch16_224')
    parser.add_argument('--hugginface_token', type=str, default='', help="Hugginface personal token")
    args = parser.parse_args()
    os.environ['GUROBI_HOME'] = args.gurobi_home
    os.environ['GRB_LICENSE_FILE'] = args.gurobi_license_file
    main(args=args)
