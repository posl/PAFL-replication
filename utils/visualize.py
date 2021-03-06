from graphviz import Digraph
# from utils.constant import get_path

def pm2pic(pm_file_path):

    init_label = "S"
    final_labels = ["N","P"]

    init_node = ""
    final_nodes = []

    with open(pm_file_path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    start_idx = 2
    endmodule_idx = lines.index("endmodule")

    edges = []
    # extract edge: (start, end, prob)
    # []s=1 -> 0.9386726078799249:(s'=2) + 0.06132739212007505:(s'=3);
    for line in lines[start_idx + 2:endmodule_idx]:
        eles = line.split("->")
        assert len(eles) == 2
        start = eles[0].split("=")[-1]
        end_ndoes = eles[1].strip(";").split(" + ")
        for node in end_ndoes:
            elements = node.split(":")
            prob = float(elements[0])
            end_node = elements[1].strip(")").split("=")[-1]
            edges.append((start, end_node, prob))

    # extract label
    # s=5|s=8|s=9|s=10|s=11|s=12|s=13|s=14|s=15;
    state2label = {}
    for line in lines[endmodule_idx + 2:]:
        if line.strip() == "":
            continue
        eles = line.split()
        label = eles[1].strip('"')
        states = eles[3].strip(';').split("|")
        for state in states:
            state = state.split("=")[-1]
            state2label[state] = label
            if label == init_label:
                assert len(states) == 1
                init_node = state
            if label in final_labels:
                assert len(states) == 1
                final_nodes.append(state)

    ##############
    #make gv
    ##############
    f = Digraph('finite_state_machine', filename='fsm.gv')
    f.attr(rankdir='LR', size='8,5')
    f.attr('node', shape='doublecircle')
    
    # make init and final node
    f.attr('node', shape='doublecircle')
    f.node(init_node.strip())
    for node in final_nodes:
        f.node(node.strip())

    # inner states
    f.attr('node', shape='circle')
    for edge in edges:
        new_label = "{:.4f}/{}".format(edge[-1], state2label[edge[1]])
        f.edge(edge[0].strip(), edge[1].strip(), label=new_label)

    f.view()

if __name__ == '__main__':
    # file_path = get_path("experiments/application/no_stopws/l2_results/lstm_mr_k2_alpha_64_107563.pm")
    file_path = "/Users/ishimotoyuuta/OneDrive/Python/rnn2automata/data/no_stopws/L2_results/mr/lstm/k=6/alpha=64/train_107644.pm"
    pm2pic(file_path)

