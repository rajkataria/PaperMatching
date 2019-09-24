import community
import csv
import datetime
import json
import matplotlib.pyplot as plt; plt.rcdefaults()
import networkx as nx
import os
import pprint
import sys

from argparse import ArgumentParser
from networkx.algorithms import bipartite

pp = pprint.PrettyPrinter(depth=6)

canonical_conflicts = {
    'fb.com': 'facebook.com',
    'uiuc.edu': 'illinois.edu'
}

def draw_bipartite_graph(G):
    X, Y = bipartite.sets(G)
    pos = dict()
    pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
    pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
    nx.draw(G, pos=pos, with_labels=True, font_size=8, node_size=1000)

    fig = plt.gcf()
    plt.tight_layout()
    fig.set_size_inches(148, 84)
    plt.savefig('bipartite-graph.png')

def draw_weighted_graph(G, fn, layout=None):
    plt.gcf().clear()
    plt.clf()
    try:
        exxlarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 8]
        exlarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 5 and d['weight'] < 8]
        elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 3 and d['weight'] < 5]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 1 and d['weight'] < 3]
    except:
        exxlarge = [(u, v) for (u, v, d) in G.edges(data=True)]
        exlarge = [(u, v) for (u, v, d) in G.edges(data=True)]
        elarge = [(u, v) for (u, v, d) in G.edges(data=True)]
        esmall = [(u, v) for (u, v, d) in G.edges(data=True)]

    if layout is None:
        pos = nx.spring_layout(G, iterations=2)  # positions for all nodes
    else:
        pos = layout(G)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=exlarge,
                           width=6, edge_color='r')

    nx.draw_networkx_edges(G, pos, edgelist=exlarge,
                           width=6, edge_color='y')

    nx.draw_networkx_edges(G, pos, edgelist=elarge,
                           width=3, edge_color='b')

    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=2, alpha=0.5, edge_color='g', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    edge_labels = []
    for u,v,d in G.edges(data=True):
        if 'attr' in d:
            edge_labels.append(((u,v),d['attr']['label']))
    edge_labels = dict(edge_labels)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels)
    fig = plt.gcf()
    plt.tight_layout()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(fn)
    plt.gcf().clear()
    plt.clf()

def normalize_conflicts(conflicts):
    normalized_conflicts = []
    for c in conflicts:
        if c.startswith('cs.'):
            c = c.replace('cs.','')
        if c.startswith('cse.'):
            c = c.replace('cse.','')
        if c in canonical_conflicts:
            c = canonical_conflicts[c]    
        normalized_conflicts.append(c)

    return list(set(normalized_conflicts))

def parse_users(users_fn):
    users = {}
    organizations = {}
    with open(users_fn, 'r') as f:
        data = f.readlines()
        for d, datum in enumerate(data):
            # Skip the header
            # First Name    Middle Initial (optional)   Last Name   E-mail  Organization    Domain Conflicts
            if d == 0:
                continue
            firstname, _, lastname, email, org, conflicts = datum.split('\t')

            conflicts_list = []
            for c in conflicts.split(';'):
                if c.strip() != '':
                    conflicts_list.append(c.strip())
            
            users['{} | {}'.format(lastname, firstname)] = \
                {'lastname': lastname, 'firstname': firstname, 'email': email, 'org': org, 'conflicts': conflicts_list}

    return users, organizations

def autofill_institution_conflicts(metareviewer):
    org = metareviewer['org']
    lastname = metareviewer['lastname']
    firstname = metareviewer['firstname']
    conflicts = []
    if 'MIT' in org:
        conflicts.append('mit.edu')
    if 'University of Pennsylvania' in org:
        conflicts.append('upenn.edu')
    if 'Nanyang Technological University' in org:
        conflicts.append('ntu.edu.sg')
    if 'UCSD' in org:
        conflicts.append('ucsd.edu')
    if 'Stony Brook University' in org:
        conflicts.append('stonybrook.edu')
    if 'UIUC' in org:
        conflicts.append('illinois.edu')
    if 'Simon Fraser University' in org:
        conflicts.append('sfu.ca')
    if 'UC Berkeley' in org:
        conflicts.append('berkeley.edu')
    
    metareviewer['conflicts'].extend(conflicts)
    if len(metareviewer['conflicts']) > 0:
        metareviewer['conflicts-entered'] = True

def conflict_exceptions(metareviewer):
    # Not the most efficient way of doing this, but exception list is small so not a big deal
    with open('conflict-exceptions.json', 'r') as fin:
        exceptions = json.load(fin)
        for e in exceptions:
            if e['firstname'].strip() == metareviewer['firstname'].strip() and  e['lastname'].strip() == metareviewer['lastname'].strip():
                if len(set(metareviewer['conflicts']).intersection(set(e['conflicts']))) == len(set(metareviewer['conflicts'])):
                    print ('\tUpdating conflicts for {} {}'.format(e['firstname'], e['lastname']))
                    metareviewer['conflicts'] = e['revised-conflicts']

def parse_metareviewers(metareviewers_fn, users, organizations):
    metareviewers = {}
    domains = []
    with open(metareviewers_fn, encoding='latin-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for d, datum in enumerate(spamreader):
            # Skip the header
            # 'First Name', 'Last Name', 'Email', 'Organization', 'Assigned', 'Completed', '% Completed', 
            # 'Bids', 'Domain Conflicts Entered', 'User Type', 'Selected', 'Primary', 'Secondary'
            if d == 0:
                continue
            firstname, lastname, email, org, assigned, comp, comp_percent, bids, conflicts_entered, user_type, sa_selected, sa_primary, sa_secondary = \
                datum
            key = '{} | {}'.format(lastname, firstname)
            metareviewers[key] = {
                'lastname': lastname, 
                'firstname': firstname, 
                'email': email, 
                'org': org,
                'original-conflicts': users[key]['conflicts'],
                'original-conflicts-entered': True if conflicts_entered == 'Yes' else False,
                'conflicts': users[key]['conflicts'], # Field name changes later
                'conflicts-entered': True if conflicts_entered == 'Yes' else False, # Field name changes later
                'assigned': assigned,
                'completed': comp,
                'completed percent': comp_percent,
                'bids': bids,
                'user type': user_type,
                'selected': sa_selected,
                'primary': sa_primary,
                'secondary': sa_secondary
                }
            
            metareviewers[key]['conflicts'] = normalize_conflicts(metareviewers[key]['conflicts'])
            
            conflict_exceptions(metareviewers[key])
            if not metareviewers[key]['conflicts-entered']:
                autofill_institution_conflicts(metareviewers[key])
            
            domains.extend(metareviewers[key]['conflicts'])    
            if not metareviewers[key]['conflicts-entered']:
                print (datum)
                print (metareviewers[key])
                print (organizations[metareviewers[key]['org']])
                sys.exit(1)
    domains = list(set(domains))
    return metareviewers, domains

def normalize_names(key):
    mapping = {
        'Schwing | Alexander': 'Schwing | Alex',
        'Taylor | Camillo' : 'Taylor | Camillo J.',
        'Wu | Ying Nian' : 'Wu | Ying',
        'Barron | Jonthan' : 'Barron | Jon',
        'Li | Li-Jia' : 'Li | Jia',
        'Efros | Alexei' : 'Efros | Alexei (Alyosha)'}
    if key in mapping:
        return mapping[key].lower()
    return key.lower()

def parse_recent_acs(recent_acs_fn, metareviewers):
    recent_acs = {}
    with open(recent_acs_fn) as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for d, datum in enumerate(reader):
            if d == 0:
                continue            
            _, _, _, _, _, _, name, num_ac, _, _, _ = datum
            names = name.split(' ')
            if len(names) == 2:
                key = '{} | {}'.format(names[1], names[0]).lower()
            elif len(names) == 3:
                key = '{} | {}'.format(names[2], ' ' .join([names[0], names[1]])).lower()
            elif len(names) == 4:
                key = '{} | {}'.format(names[3], ' ' .join([names[0], names[1], names[2]])).lower()
            else:
                print ('Unhandled case "{}"...Exiting'.format(names))


            recent_acs[key] = int(num_ac)

    for key in metareviewers:
        k = normalize_names(key)
        if k not in recent_acs:
            print ('\tReviewer {} has not been an AC before'.format(key))
            metareviewers[key]['# Times AC'] = 0
        else:
            metareviewers[key]['# Times AC'] = recent_acs[k]

def formulate_bipartite_graph(metareviewers, domains, options):
    graph = nx.Graph()
    graph.add_nodes_from(metareviewers.keys(), bipartite=0)
    graph.add_nodes_from(domains, bipartite=1)
    for m in metareviewers:
        graph.add_edges_from([(m,c) for c in metareviewers[m]['conflicts']])

    c = nx.algorithms.bipartite.clustering(graph, metareviewers.keys())

    pp.pprint('#'*100)
    pp.pprint('Total MetaReviewers: {}'.format(len(metareviewers.keys())))
    pp.pprint(domains)
    pp.pprint ('='*100)
    pp.pprint(c)

def make_graph(arranged_graphs, graph_size, level, debug):
    if graph_size in arranged_graphs:
        g = arranged_graphs[graph_size].pop()
        if len(arranged_graphs[graph_size]) == 0:
            del arranged_graphs[graph_size]
        return [g]
    else:
        if debug:
            print ('\t'*level + 'Making graph of size: {} / {}'.format(graph_size-1, 1))

        g1 = make_graph(arranged_graphs, graph_size-1, level+1, debug)
        g2 = make_graph(arranged_graphs, 1, level+1, debug)
        if debug:
            for g in g1:
                print ('\t'*level + 'Graphs created of size: {}'.format(len(g.nodes())))
            for g in g2:
                print ('\t'*level + 'Graphs created of size: {}'.format(len(g.nodes())))
        g1.extend(g2)
        return g1

def restore_edges(G, F):
    for n1,n2 in F.edges():
        if n1 in G and n2 in G and not G.has_edge(n1,n2):
            G.add_weighted_edges_from([(n1, n2, F[n1][n2]['weight'])])

def merge_graphs(full_graph, graphs, options):
    print('Merge processed graphs to appropriate size')
    debug = False
    arranged_graphs = {}
    final_graphs = []
    group_size = int(1.0*options['num_reviewers']/options['num_groups'])
    sorted_lengths = []
    for g in graphs:
        key = len(g.nodes())
        sorted_lengths.append(key)
        if key in arranged_graphs:
            arranged_graphs[key].append(g)
        else:
            arranged_graphs[key] = [g]
    
    sorted_lengths = sorted(sorted_lengths, reverse=True)

    for jj in arranged_graphs:
        print('\t{} graphs with length {}'.format(len(arranged_graphs[jj]),jj ))
    
    print ('#'*100)
    print ('#'*100)
    counter = 0
    while True:
        k = sorted_lengths[counter]
        counter += 1
        if len(arranged_graphs.keys()) == 0:
            break

        if k not in arranged_graphs:
            continue

        print ('\tStarting graph size: {}'.format(k))
        

        g = arranged_graphs[k].pop()
        if len(arranged_graphs[k]) == 0:
            del arranged_graphs[k]

        merged_graphs = make_graph(arranged_graphs, group_size - len(g.nodes()), level=2, debug=debug)
        merged_graphs.append(g)
        for m in merged_graphs:
            print ('\tMerged graphs: {}'.format(len(m.nodes())))
        
        print ('\t' + '!'*100)
        final_graphs.append(merged_graphs)
        
    
    print ('Final check of graph sizes')
    for ii, f in enumerate(final_graphs):
        counter = 0
        for g in f:
            counter += len(g.nodes())
        print ('\tGraph {} of size: {}'.format(ii, counter))
    
    combined_graphs = []
    final_partition = {}
    for ii, g in enumerate(final_graphs):
        plt.gcf().clear()
        plt.clf()
        combined_graph = nx.algorithms.operators.all.compose_all(g)
        combined_graphs.append(combined_graph)
        restore_edges(combined_graph, full_graph)
        draw_weighted_graph(combined_graph, fn='weighted-graph-{}.png'.format(ii), layout=nx.shell_layout)
        for n in combined_graph.nodes():
            final_partition[n] = ii

    graph_union = nx.algorithms.operators.all.compose_all(combined_graphs)
    print ('Original graph: {} edges. Combined graph: {} edges'.format(len(full_graph.edges()), len(graph_union.edges())))
    draw_weighted_graph(graph_union, fn='combined-weighted-graphs.png', layout=nx.shell_layout)
    draw_weighted_graph(full_graph, fn='full-graph.png', layout=nx.shell_layout)

    community_graph = community.induced_graph(final_partition, full_graph, weight='weight')
    draw_weighted_graph(community_graph, fn='community_graph.png', layout=nx.shell_layout)
    graph_difference = nx.algorithms.operators.binary.difference(full_graph, graph_union)

    remove_nodes = []
    isolates = nx.isolates(graph_difference)
    for n in isolates:
        remove_nodes.append(n)
    graph_difference.remove_nodes_from(remove_nodes)
    draw_weighted_graph(graph_difference, fn='missing-edges.png', layout=nx.shell_layout)
    return final_partition

def find_between_community_edges(partition, G):
    print (partition)
    print ('*'*100)
    edges = {}
    edges_count = {}

    for (ni, nj) in G.edges():
        ci = int(partition[ni])
        cj = int(partition[nj])
        if cj < ci:
            ci,cj = cj,ci
        if ci != cj:
            try:
                edges[(ci, cj)].append([(ni, nj)])
            except KeyError:
                edges[(ci, cj)] = [(ni, nj)]

    community_edges = []
    for c_e in edges:
        c1,c2 = c_e
        community_edges.append([len(edges[c_e]), c1, c2])
        edges_count[c_e] = len(edges[c_e])

    community_edges = sorted(community_edges,key=lambda x: x[0], reverse=True)
    return edges, edges_count, community_edges#, sorted(edges, key=len, reverse=True)

def get_partition_sizes(partition):
    partition_sizes = {}
    for ii,name in enumerate(partition):
        partition_num = partition[name]
        if partition_num not in partition_sizes:
            partition_sizes[partition_num] = 0
        partition_sizes[partition_num] += 1
    return partition_sizes

def aggregate_partitions(partition, G, group_size):
    print ('Aggregate partitions')
    complete = False
    while not complete:
        merged_partitions = False
        partition_sizes = get_partition_sizes(partition)
        edges, edges_count, community_edges = find_between_community_edges(partition, G)

        for count,ci,cj in community_edges:
            if partition_sizes[ci] + partition_sizes[cj] > group_size:
                continue
            merged_partitions = True
            for name in partition:
                if partition[name] == cj:
                    partition[name] = ci
            break
        if merged_partitions:
            continue
        complete = True
    return partition
    # sys.exit(1)
def formulate_weighted_graph(metareviewers, domains, options):
    edges = []
    graph = nx.Graph()
    graph.add_nodes_from(metareviewers.keys(), bipartite=0)
    
    for i, r1 in enumerate(sorted(metareviewers.keys())):
        for j, r2 in enumerate(sorted(metareviewers.keys())):
            if i <= j:
                continue
            domain_conflicts = list(set(metareviewers[r1]['conflicts']).intersection(set(metareviewers[r2]['conflicts'])))
            edge_weight = len(domain_conflicts)
            if edge_weight > 0:
                graph.add_weighted_edges_from([(r1, r2, edge_weight)], attr={'label': ';'.join(domain_conflicts)})

    group_size = int(1.0*options['num_reviewers']/options['num_groups'])
    print ('Ideal cluster size: {}'.format(group_size))


    remaining_subgraphs = []
    processed_subgraphs = []
    for k,cc_nodes in enumerate(sorted(nx.connected_components(graph), key=len, reverse=True)):
        remaining_subgraphs.append(graph.subgraph(cc_nodes))


    partition_counter = 0
    while len(remaining_subgraphs) > 0:
        partition_counter += 1
        cc = remaining_subgraphs.pop()
        if len(cc) > group_size:
            from networkx.algorithms.community import greedy_modularity_communities, asyn_fluidc, girvan_newman, kernighan_lin_bisection, k_clique_communities, label_propagation_communities

            partition = community.best_partition(cc)
            partition = aggregate_partitions(partition, cc, group_size)

            partitioned_nodes = {}
            partitioned_subgraphs = []
            for ii, m in enumerate(partition):
                if partition[m] not in partitioned_nodes:
                    partitioned_nodes[partition[m]] = [m]
                else:
                    partitioned_nodes[partition[m]].append(m)
            for p in partitioned_nodes:
                partitioned_subgraphs.append(graph.subgraph(partitioned_nodes[p]))
            remaining_subgraphs.extend(partitioned_subgraphs)
        else:
            processed_subgraphs.append(cc)

    print ('Processed subgraphs: {}'.format(len(processed_subgraphs)))
    partition = merge_graphs(graph, processed_subgraphs, options)
    return partition

def output_csvs(metareviewers, partition):
    group_conflicts = {}
    all_institutions = []
    with open('panel_assignments.csv', 'w') as csvfile:
        fieldnames = ['Group', 'Last name', 'First name', 'Org', 'Conflicts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for p in range(0,max(partition.values()) + 1):
            group_conflicts[p] = []
            for name in partition:
                if partition[name] != p:
                    continue
                metareviewer = metareviewers[name]
                print ('{},{},{},{},{}'.format(p,metareviewer['lastname'],metareviewer['firstname'], metareviewer['org'], metareviewer['conflicts']))
                writer.writerow({
                    'Group': p,
                    'Last name': metareviewer['lastname'],
                    'First name': metareviewer['firstname'], 
                    'Org': metareviewer['org'],
                    'Conflicts': ' ; '.join(metareviewer['conflicts'])
                    })
                group_conflicts[p].extend(metareviewer['conflicts'])

            print('{},,,,'.format(set(group_conflicts[p])))
            writer.writerow({
                    'Group': 'Group conflicts: {}'.format(set(group_conflicts[p])),
                    'Last name': '',
                    'First name': '', 
                    'Org': '',
                    'Conflicts': ''
                    })

            print(',,,,')
            writer.writerow({
                    'Group': '',
                    'Last name': '',
                    'First name': '', 
                    'Org': '',
                    'Conflicts': ''
                    })
            all_institutions.extend(group_conflicts[p])
        
    with open('institutional_assignments.csv', 'w') as csvfile:
        fieldnames = ['Institution', 'Group(s)', 'Total groups']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for inst in sorted(set(all_institutions)):
            groups = []
            for p in range(0,max(partition.values())):
                if inst in group_conflicts[p]:
                    groups.append(str(p))
            writer.writerow({
                    'Institution': inst,
                    'Group(s)': ' ; '.join(groups),
                    'Total groups': len(groups)
                    })

    with open('metareviewers-aggregated.csv', 'w') as csvfile:
        fieldnames = ['First Name', 'Last Name', 'Email', 'Organization', 'Assigned', 'Completed', '% Completed', 'Bids', \
            'Domain Conflicts Entered', 'Revised Domain Conflicts Entered', 'User Type', 'Subject Areas - Selected', 'Subject Areas - Primary', \
            'Subject Areas - Secondary', 'Conflicts', 'Revised Conflicts', 'Group', '# Times AC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key in metareviewers:
            m = metareviewers[key]

            writer.writerow({
                    'First Name': m['firstname'],
                    'Last Name': m['lastname'],
                    'Email': m['email'],
                    'Organization': m['org'], 
                    'Assigned': m['assigned'], 
                    'Completed': m['completed'], 
                    '% Completed': m['completed percent'],
                    'Bids': m['bids'],
                    'Domain Conflicts Entered': m['original-conflicts-entered'], 
                    'Revised Domain Conflicts Entered': m['conflicts-entered'], 
                    'User Type': m['user type'], 
                    'Subject Areas - Selected': m['selected'], 
                    'Subject Areas - Primary': m['primary'], 
                    'Subject Areas - Secondary': m['secondary'],
                    'Conflicts': m['original-conflicts'], 
                    'Revised Conflicts': m['conflicts'],
                    'Group': partition[key],
                    '# Times AC': m['# Times AC']
                    })            

def main():
    parser = ArgumentParser(description='')
    parser.add_argument('-u', '--users_txt', help='Users.txt as exported from CMT (contains conflict information)')
    parser.add_argument('-m', '--metareviewers_csv', help='metareviewers.csv as copied and pasted from CMT')
    parser.add_argument('-r', '--recent_acs_csv', help='Recents ACs csv file')
    parser.add_argument('-p', '--partition', help='partition file')
    parser_options = parser.parse_args()

    users, organizations = parse_users(parser_options.users_txt)
    metareviewers, domains = parse_metareviewers(parser_options.metareviewers_csv, users, organizations)
    parse_recent_acs(parser_options.recent_acs_csv, metareviewers)

    options = {'num_groups': 8, 'num_reviewers': len(metareviewers)}
    if parser_options.partition is None:
        partition = formulate_weighted_graph(metareviewers, domains, options)
        with open('partition-{}.json'.format(str(datetime.datetime.now())), 'w') as pout:
            json.dump(partition, pout, indent=4, sort_keys=True)
    else:
        with open(parser_options.partition, 'r') as pin:
            partition = json.load(pin)

    output_csvs(metareviewers, partition)
    
if __name__ == '__main__':
    main()