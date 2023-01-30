##############################################################################################################
##############################################################################################################
# Created by Rajbir Kataria (UIUC) - rk2@illinois.edu
##############################################################################################################
##############################################################################################################
import copy
import csv
import json
import matplotlib.pyplot as plt; plt.rcdefaults()
import networkx as nx
import networkx.algorithms.flow as flow
import numpy as np
import os
import shutil
import statistics
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from argparse import ArgumentParser
from timeit import default_timer as timer
from ortools.graph.python import min_cost_flow
from timeit import default_timer as timer

experience_scores = {
    'Researcher/faculty': 1.0,
    'Student': 1.0,
    '': 1.0
}

canonical_conflicts = {
    'fb.com': 'facebook.com',
    'uiuc.edu': 'illinois.edu'
}

def normalize_conflicts(conflicts):
    normalized_conflicts = []
    for c in conflicts:
        if c.startswith('cs.'):
            c = c.replace('cs.','')
        if c.startswith('cse.'):
            c = c.replace('cse.','')
        if c.startswith('eecs.'):
            c = c.replace('eecs.','')
        if c.startswith('ece.'):
            c = c.replace('ece.','')
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
            firstname, _, lastname, email, org, country, conflicts = datum.split('\t')

            conflicts_list = []
            for c in conflicts.split(';'):
                if c.strip() != '':
                    conflicts_list.append(c.strip())
            key = '{}'.format(email)
            users[key] = \
                {'lastname': lastname, 'firstname': firstname, 'email': email, 'org': org, 'conflicts': conflicts_list, 'country': country}

            users[key]['conflicts'] = normalize_conflicts(users[key]['conflicts'])

    print ('Total users: {}'.format(len(users.keys())))
    return users, organizations

def parse_conflicts(conflicts_fn, debug=False):
    paper_conflicts = {}
    total_conflicts = 0
    with open(conflicts_fn) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for d, datum in enumerate(spamreader):
            # Skip the header
            # Paper ID    Reviewer Email
            if d == 0:
                continue

            paper_id, reviewer_email = datum
            key = '{}'.format(paper_id)
            if key not in paper_conflicts:
                paper_conflicts[key] = []

            paper_conflicts[key].append(reviewer_email)
            total_conflicts += 1            
    print ('Total conflicts: {}'.format(total_conflicts))
    return paper_conflicts

def parse_reviewers(reviewers_fn, users, organizations, debug=False):
    reviewers = {}
    domains = []
    with open(reviewers_fn) as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for d, datum in enumerate(spamreader):
            # Skip the header
            # 'First Name', 'Last Name', 'Email', 'Organization', 'Quota', 'Assigned', 'Completed', '% Completed',
            # 'Bids', 'Domain Conflicts Entered', 'User Type', 'External Profile Entered', 'Selected', 'Primary', 'Secondary', 'Actions'
            
            if d == 0:
                continue
            firstname, lastname, email, org, quota, assigned, comp, comp_percent, bids, conflicts_entered, user_type, \
            ext_profile_entered, sa_selected, sa_primary, sa_secondary, _ = datum
            key = '{}'.format(email.strip())
            if key in reviewers:
                if debug:
                    print ('\tUser already exists: {}'.format(key))
                continue

            # quota can't be cast to int if '-' is included for unavailable quotas
            reviewers[key] = {
                'lastname': lastname.strip(),
                'firstname': firstname.strip(),
                'email': email.strip(),
                'org': org.strip(),
                'quota': quota,
                'original-conflicts': users[key]['conflicts'],
                'original-conflicts-entered': conflicts_entered == 'Yes',
                'conflicts': users[key]['conflicts'],
                'conflicts-entered': conflicts_entered.strip() == 'Yes', # Field name changes later
                'assigned': assigned,
                'completed': comp,
                'completed percent': comp_percent,
                'bids': bids,
                'user type': user_type.strip(),
                'external_profile_entered': ext_profile_entered,
                'selected': sa_selected.strip(),
                'primary': sa_primary.strip(),
                'secondary': sa_secondary.strip(),
                'papers': {}
                }
                        
            domains.extend(reviewers[key]['conflicts'])    
            if not reviewers[key]['conflicts-entered']:
                print ('\tDomain conflicts not entered for the reviewer: {}'.format(key))
                
    domains = list(set(domains))

    print ('Total reviewers: {}'.format(len(reviewers.keys())))
    return reviewers, domains

def parse_papers(papers_fn):
    papers = {}
    with open(papers_fn, encoding='latin-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for d, datum in enumerate(reader):
            # Skip the header
            # 'Paper ID', 'Created', 'Last Modified', 'Paper Title', 'Abstract', 'Author Names', 'Author Emails', 
            # 'Track Name', 'Primary Subject Area', 'Secondary Subject Areas', 'Conflicts', 'Assigned', 'Completed', 
            # '% Completed', 'Bids', 'Discussion', 'Status', 'Requested For Camera Ready', 'Camera Ready Submitted?', 
            # 'Requested For Author Feedback', 'Author Feedback Submitted?', 'Files', 'Number of Files', 
            # 'Supplementary Files', 'Number of Supplementary Files', 'Reviewers', 'Reviewer Emails', 'MetaReviewers', 
            # 'MetaReviewer Emails', 'SeniorMetaReviewers', 'SeniorMetaReviewerEmails', 
            # 'Q3 (Submission Checklist)', 'Q4 (Policies Concerning Plagiarism and Double Submissions)', 'Q5 (Policy Concerning Media)'

            if d == 0:
                continue
            paper_id, created, modified, title, abstract, author_names, author_emails, track_name, primary_sa, secondary_sa, \
                conflicts, assigned, completed, percent_completed, bids, discussion, status, requested_camera_ready, camera_ready_submitted, \
                requested_author_feedback, author_feedback_submitted, files, num_files, supplementary_files, num_supplementary_files, \
                reviewers, reviewer_emails, metareviewers, metareviewer_emails, sr_metareviewers, sr_metareviewer_emails, q3, q4, q5 = datum
            
            if status != 'Awaiting Decision':
                continue

            papers[str(paper_id)] = {
                'Paper ID': paper_id, 
                'Created': created, 
                'Last Modified': modified, 
                'Paper Title': title, 
                'Abstract': abstract, 
                'Author Names': author_names, 
                'Author Emails': author_emails, 
                'Track Name': track_name, 
                'Primary Subject Area': primary_sa,
                'Secondary Subject Areas': secondary_sa, 
                'Conflicts': conflicts, 
                'Assigned': assigned, 
                'Completed': completed, 
                '% Completed': percent_completed, 
                'Bids': bids, 
                'Discussion': discussion, 
                'Status': status, 
                'Requested For Camera Ready': requested_camera_ready, 
                'Camera Ready Submitted?': camera_ready_submitted, 
                'Requested For Author Feedback': requested_author_feedback, 
                'Author Feedback Submitted?': author_feedback_submitted, 
                'Files': files, 
                'Number of Files': num_files, 
                'Supplementary Files': supplementary_files, 
                'Number of Supplementary Files': num_supplementary_files, 
                'Reviewers': reviewers, 
                'Reviewer Emails': reviewer_emails, 
                'MetaReviewers': metareviewers, 
                'MetaReviewer Emails': metareviewer_emails, 
                'SeniorMetaReviewers': sr_metareviewers, 
                'SeniorMetaReviewerEmails': sr_metareviewer_emails,
                'Q3 (Submission Checklist)': q3,
                'Q4 (Policies Concerning Plagiarism and Double Submissions)': q4,
                'Q5 (Policy Concerning Media)': q5,
                'reviewers': {}
            }
    print ('Total papers: {}'.format(len(papers.keys())))
    return papers

def parse_reviewer_suggestions(reviewer_suggestions_fn):
    reviewer_suggestions = {}
    with open(reviewer_suggestions_fn, encoding='latin-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for d, datum in enumerate(spamreader):
            # Skip the header
            # Paper ID  Meta-Reviewer Email Reviewer Email  Rank
            if d == 0:
                continue
            paper_id, metareviewer_email, reviewer_email, rank = datum

            if paper_id not in reviewer_suggestions:
                reviewer_suggestions[paper_id] = {
                    'Paper ID': paper_id, 
                    'MetaReviewer': metareviewer_email, 
                    'Reviewers': [reviewer_email]
                }
            else:
                reviewer_suggestions[paper_id]['Reviewers'].append(reviewer_email)
    print ('Total paper suggestions: {}'.format(len(reviewer_suggestions.keys())))
    return reviewer_suggestions

def calculate_subject_area_score(primary_paper, secondary_paper, primary_reviewer, secondary_reviewer):
    score = 0.0
    if primary_paper == primary_reviewer:
        score += 0.6

    if primary_paper in secondary_reviewer:
        score += 0.4

    num_common_secondary_sas = len(set(secondary_paper.split(';')).intersection(set(secondary_reviewer.split(';'))))
    num_paper_secondary_sas = len(set(secondary_paper.split(';')))

    score += 0.4 * num_common_secondary_sas/ max(num_paper_secondary_sas, 1)
    return score

def calculate_suggestion_score(paper, reviewer_email):
    if reviewer_email not in paper['Reviewers']:
        score = 0.0
    else:
        index = paper['Reviewers'].index(reviewer_email)
        if index <= 19:
            score = (20.0 - index) / 20.0
        else:
            score = 1.0 / 20.0
    return score

def parse_tpms_scores_file(tpms_fn):
    tpms_scores = {}
    with open(tpms_fn, encoding='latin-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for d, datum in enumerate(spamreader):
            # Skip the header
            # 'Paper ID', 'Email', 'TPMS Score'
            if d == 0:
                continue
            paper_id, email, tpms = datum
            paper_id, email, tpms = paper_id.strip(), email.strip(), float(tpms)

            if email not in tpms_scores:
                tpms_scores[email] = {}
            tpms_scores[email][paper_id] = tpms

    return tpms_scores

def parse_and_calculate_scores(tpms_scores_fn, reviewers, papers, reviewer_suggestions, w_t, w_a, w_s, w_e, debug=False):
    tpms_scores = parse_tpms_scores_file(tpms_scores_fn)
    for email in reviewers:
        for paper_id in papers:
            if email not in reviewers:
                continue
                
            # Since papers are filtered based on "Awaiting Decision"
            if paper_id not in papers:
                continue 

            initial_scores = {'tpms_scores': 0.0, 'subject_area_scores': 0.0, 'suggestion_scores': 0.0, 'experience_scores': 0.0, 'final_scores': 0.0, 'conflicts': []}

            papers[paper_id]['reviewers'][email] = initial_scores
            reviewers[email]['papers'][paper_id] = initial_scores
            
            subject_area_score = calculate_subject_area_score(papers[paper_id]['Primary Subject Area'], papers[paper_id]['Secondary Subject Areas'],\
                reviewers[email]['primary'], reviewers[email]['secondary'])

            if paper_id in reviewer_suggestions:
                suggestion_score = calculate_suggestion_score(reviewer_suggestions[str(paper_id)], email)
            else:
                if debug:
                    print ('\t\tNo suggestions for paper: {}. Assigning a suggestion score of 0.0'.format(paper_id))
                suggestion_score = 0.0

            tpms_score = 0.0
            if email in tpms_scores and paper_id in tpms_scores[email]:
                tpms_score = tpms_scores[email][paper_id]
            papers[paper_id]['reviewers'][email]['tpms_scores'] = tpms_score
            reviewers[email]['papers'][paper_id]['tpms_scores'] = tpms_score

            papers[paper_id]['reviewers'][email]['subject_area_scores'] = subject_area_score
            reviewers[email]['papers'][paper_id]['subject_area_scores'] = subject_area_score

            papers[paper_id]['reviewers'][email]['suggestion_scores'] = suggestion_score
            reviewers[email]['papers'][paper_id]['suggestion_scores'] = suggestion_score

            final_score = w_t * tpms_score + w_a * subject_area_score + w_s * suggestion_score + w_e * experience_scores[reviewers[email]['user type']]

            papers[paper_id]['reviewers'][email]['final_scores'] = final_score
            reviewers[email]['papers'][paper_id]['final_scores'] = final_score


def prettify(elem):
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

def save_xml_assignments(output_folder, assignment_matrix, reviewer_mapping, paper_mapping):
    assignments = ET.Element('assignments')  
    for i in range(0, assignment_matrix.shape[0]):
        for j in range(0, assignment_matrix.shape[1]):
            if assignment_matrix[i,j] == 0:
                continue
            paperid = paper_mapping[str(j)]
            reviewerid = reviewer_mapping[str(i)]
            submission = ET.SubElement(assignments, 'submission') 
            submission.set('submissionId', paperid)
            user = ET.SubElement(submission, 'user') 
            user.set('email', reviewerid)
    with open(os.path.join(output_folder, 'assignments.xml'), 'w') as fout:
        fout.write(prettify(assignments))

def save(data, fn):
    with open(fn, 'w') as fout:
        json.dump(data, fout, indent=4, sort_keys=True)

def load(fn):
    with open(fn, 'r') as fin:
        data = json.load(fin)
    return data

def parse_authors(s_authors):
    authors = s_authors.split(';')
    for i,a in enumerate(authors):
        authors[i] = authors[i].replace('*','').strip()
    return authors

def get_authors_and_reviewers_with_coauthors(reviewers, papers):
    authors = []
    coauthors = {}
    authors_and_reviewers_with_coauthors = {}
    for p in papers:
        paper_authors = parse_authors(papers[p]['Author Emails'])
        authors.extend(paper_authors)
        for pa in paper_authors:
            if pa not in coauthors:
                coauthors[pa] = []
            coauthors[pa].extend(paper_authors)

    authors_and_reviewers = list(set(authors).intersection(set(reviewers.keys())))
    for c in coauthors:
        if c in authors_and_reviewers:
            authors_and_reviewers_with_coauthors[c] = list(set(coauthors[c]))
    return authors_and_reviewers_with_coauthors

def formulate_matrices(reviewers, reviewer_capacities, paper_conflicts, papers, users, debug=False):
    scores_matrix = np.zeros((len(reviewers.keys()), len(papers.keys())))
    tpms_scores_matrix = np.zeros((len(reviewers.keys()), len(papers.keys())))
    subject_area_scores_matrix = np.zeros((len(reviewers.keys()), len(papers.keys())))
    suggestion_scores_matrix = np.zeros((len(reviewers.keys()), len(papers.keys())))
    experience_scores_matrix = np.zeros((len(reviewers.keys()), len(papers.keys())))

    conflicts_matrix = np.zeros(scores_matrix.shape)
    capacity_vector = np.zeros(scores_matrix.shape[0]).astype(np.int)
    reviewer_adjusted_capacities = []  # reviewer-calculated capacity pairs
    reviewer_mapping = {}
    paper_mapping = {}
    reverse_reviewer_mapping = {}
    reverse_paper_mapping = {}

    for i,r in enumerate(sorted(reviewers.keys())):
        reviewer_mapping[str(i)] = r
        reverse_reviewer_mapping[r] = str(i)
        capacity_vector[i] = reviewer_capacities[reviewers[r]['user type']]

        # need to filter out quotas which are '-' and maybe 0?
        # Overwrite capacity based on user type with entered quota
        if reviewers[r]['quota'] != '' and int(reviewers[r]['quota']) < capacity_vector[i]:
            capacity_vector[i] = int(reviewers[r]['quota'])

        if 'Emergency' in reviewers[r]['secondary']:
            if debug:
                print ("Emergency reviewer - {}. Decrementing capacity by 1".format(r))

            capacity_vector[i] = capacity_vector[i] - 1

        # Record adjusted capacity for reviewer (after factoring in quota/user type/emergency reviewer
        reviewer_adjusted_capacities.append([r, capacity_vector[i]])

        for j,p in enumerate(sorted(papers.keys())):
            paper_mapping[str(j)] = str(p)
            reverse_paper_mapping[str(p)] = str(j)
            scores_matrix[i,j] = reviewers[r]['papers'][p]['final_scores']
            tpms_scores_matrix[i,j] = reviewers[r]['papers'][p]['tpms_scores']
            subject_area_scores_matrix[i,j] = reviewers[r]['papers'][p]['subject_area_scores']
            suggestion_scores_matrix[i,j] = reviewers[r]['papers'][p]['suggestion_scores']
            experience_scores_matrix[i,j] = reviewers[r]['papers'][p]['experience_scores']

    for p in paper_conflicts.keys():
        for r in paper_conflicts[p]:
            if p not in reverse_paper_mapping:
                # Paper not "Awaiting Decision"
                continue

            if r not in reverse_reviewer_mapping:
                print ('\t\tReviewer "{}" does not appear in reviewer list. Most likely out-of-sync files.'.format(r))
                continue

            conflicts_matrix[int(reverse_reviewer_mapping[r]), int(reverse_paper_mapping[p])] = 1

    authors_and_reviewers_with_coauthors = get_authors_and_reviewers_with_coauthors(reviewers, papers)
    
    for i,r in enumerate(sorted(reviewers.keys())):
        for j,p in enumerate(sorted(papers.keys())):
            common_coauthors = []

            # Raj: don't assign this to 0
            conflict = conflicts_matrix[i, j]
            authors = parse_authors(papers[p]['Author Emails'])
            author_conflicts = []
            for a in authors:
                author_conflicts.extend([c.strip() for c in users[a]['conflicts']])
            author_conflicts = list(set(author_conflicts))
            for author_conflict in author_conflicts:
                if author_conflict in reviewers[r]['conflicts']:
                    conflict = 1

            
            if r in authors_and_reviewers_with_coauthors:
                common_coauthors = set(authors_and_reviewers_with_coauthors[r]).intersection(set(authors))

            if conflicts_matrix[i,j] == 0 and conflict == 1:
                print ('\t\tFound additional conflicts(based on domains): Authors:{}, Paper:{} conflicts({}) with Reviewer: {}'.format(\
                    papers[p]['Author Emails'], p, set(reviewers[r]['conflicts']).intersection(author_conflicts), r
                    ))

            if len(common_coauthors) > 0 and conflicts_matrix[i,j] == 0:
                print ('\t\tFound coauthor conflict: Reviewer: {} has common coauthors: {} with Paper: {}'.format(r, common_coauthors, p))

            if len(common_coauthors) > 0:
                conflict = 1
                
            conflicts_matrix[i,j] = conflict

    return scores_matrix, \
        tpms_scores_matrix, \
        subject_area_scores_matrix, \
        suggestion_scores_matrix, \
        experience_scores_matrix, \
        conflicts_matrix, \
        capacity_vector, \
        reviewer_adjusted_capacities, \
        reviewer_mapping, \
        paper_mapping, \
        reverse_reviewer_mapping, \
        reverse_paper_mapping

def solve_assigment_problem_networkx(scores_matrix, conflicts_matrix, capacity_vector, num_reviews, debug=False):
    G = nx.DiGraph()
    G.add_node('Source', demand=-scores_matrix.shape[1]*num_reviews)
    G.add_node('Destination', demand=0)

    for i in range(0,scores_matrix.shape[0]):
        G.add_node('Reviewer-{}'.format(i),demand=0)#-capacity_matrix[i,0])
    for j in range(0,scores_matrix.shape[1]):
        G.add_node('Paper-{}'.format(j),demand=num_reviews)

    for i in range(0,scores_matrix.shape[0]):
        G.add_edge('Source', 'Reviewer-{}'.format(i), weight=0.0, capacity=1.0*capacity_vector[i])
        for j in range(0,scores_matrix.shape[1]):
            if conflicts_matrix[i,j] == 0:
                G.add_edge('Reviewer-{}'.format(i), 'Paper-{}'.format(j), weight=-scores_matrix[i,j], capacity=1)
            G.add_edge('Paper-{}'.format(j), 'Destination', weight=0.0, capacity=0)

    assignments = flow.min_cost_flow(G, demand='demand', capacity='capacity', weight='weight')

    if debug:
        print ('*'*50 + ' Result ' + '*'*50)
        print (json.dumps(assignments, indent=4, sort_keys=True))
    
    assignment_matrix = np.zeros(scores_matrix.shape)

    for r in sorted(assignments.keys()):
        if 'Reviewer' not in r:
            continue
        i = int(r.split('-')[-1])
        for p in sorted(assignments[r].keys()):
            j = int(p.split('-')[-1])
            assignment_matrix[i, j] = int(assignments[r][p])
    return assignment_matrix

def solve_assigment_problem_or_tools(scores_matrix, conflicts_matrix, capacity_vector, num_reviews, debug=False):
    assigments = {}
    assignment_matrix = np.zeros(scores_matrix.shape)
    start_nodes, end_nodes, capacities, unit_costs = [], [], [], []
    source_node = 0
    sink_node = 1 + scores_matrix.shape[0] + scores_matrix.shape[1]
    node_offset = scores_matrix.shape[0] + 1
    # Demand/supply for each node
    supplies = [num_reviews * scores_matrix.shape[1]] + [0] * scores_matrix.shape[0] + [-num_reviews]*scores_matrix.shape[1] #+ [0]

    for i in range(0,scores_matrix.shape[0]):
        start_nodes.append(source_node)
        end_nodes.append(i + 1)
        capacities.append(int(capacity_vector[i]))
        unit_costs.append(0)

    for i in range(0,scores_matrix.shape[0]):
        for j in range(0,scores_matrix.shape[1]):
            if conflicts_matrix[i,j] == 0:
                start_nodes.append(i + 1)
                end_nodes.append(j + node_offset)
                capacities.append(1)
                unit_costs.append(-int(1000.0*scores_matrix[i,j]))

    # Instantiate a SimpleMinCostFlow solver.
    smcf = min_cost_flow.SimpleMinCostFlow()
    # Add each arc.
    for i in range(0, len(start_nodes)):
        smcf.add_arcs_with_capacity_and_unit_cost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

    # Add node supplies.
    for i in range(0, len(supplies)):
        smcf.set_node_supply(i, supplies[i])

    if smcf.solve() == smcf.OPTIMAL:
        for i in range(smcf.NumArcs()):
            cost = smcf.flow(i) * smcf.unit_cost(i)

            start = smcf.tail(i)
            end = smcf.head(i)
            flow = smcf.flow(i)

            if start > 0:
                assignment_matrix[start-1, end-node_offset] = flow
    else:
        print('There was an issue with the min cost flow input.')
    return assignment_matrix

def solve_assigment_problem_greedy(scores_matrix, conflicts_matrix, capacity_vector, debug=False):
    assignment_matrix = solve_assigment_problem_or_tools(scores_matrix, conflicts_matrix, capacity_vector, num_reviews=1, debug=debug)
    capacity_vector = capacity_vector - np.sum(assignment_matrix, axis=1)
    conflicts_matrix = conflicts_matrix + assignment_matrix
    return assignment_matrix, conflicts_matrix, capacity_vector

def validate_results(output_folder, reviewer_suggestions, p_assignment_matrix, assignment_matrix, scores_matrix, \
    tpms_scores_matrix, subject_area_scores_matrix, suggestion_scores_matrix, experience_scores_matrix, conflicts_matrix, capacity_vector, \
    reviewer_mapping, paper_mapping, reverse_reviewer_mapping, num_reviews):
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    print ('*'*50 + ' Validating results ' + '*'*50)
    reviews_per_paper = np.sum(assignment_matrix, axis=0)
    reviews_per_reviewer = np.sum(assignment_matrix, axis=1)
    print ('\tTotal reviewers: {}  \t\t\t\tTotal papers: {}'.format(assignment_matrix.shape[0], assignment_matrix.shape[1]))
    print ('\tTotal reviews assigned: {}  \t\tExpected reviews assigned: {}'.format(np.sum(reviews_per_paper), num_reviews*assignment_matrix.shape[1]))

    print ('\tTotal assigments: {}'.format(np.sum(reviews_per_reviewer)))
    print ('\tMax # assigned to reviewer: {}'.format(np.max(reviews_per_reviewer)))
    print ('\tMin # assigned to reviewer: {}'.format(np.min(reviews_per_reviewer)))
    print ('\tTotal cost: {}'.format(np.sum(np.sum(scores_matrix * assignment_matrix))))
    print ('\tConflict assigments: {}'.format(np.sum(np.sum(conflicts_matrix * assignment_matrix))))

    # Reviewer paper workload
    plt.hist(reviews_per_reviewer, bins=11, range=(0,11))
    plt.xlim(0,11)
    plt.ylim(0,1000)
    plt.xlabel('# of papers/reviewers')
    plt.ylabel('# of reviewers')
    plt.title('Reviewer assignment distribution')
    plt.savefig(os.path.join(output_folder, "o-reviewer-distribution.png"))

    # Max score of 1st/2nd/3rd reviewer
    plt.clf()
    plt.hist(np.amax(assignment_matrix * scores_matrix, axis=0), bins=25, range=(0,13.0))
    plt.xlim(0,13.0)
    plt.ylim(0,4000)
    plt.xlabel('best scores')
    plt.ylabel('# of reviews')
    plt.title('Best score of papers')

    plt.savefig(os.path.join(output_folder,"o-max-scores-per-paper.png"))

    # Mean score of 1st/2nd/3rd reviewer
    plt.clf()
    plt.hist((np.sum(assignment_matrix * scores_matrix, axis=0))/num_reviews, bins=25, range=(0,13.0))
    plt.xlim(0,13.0)
    plt.ylim(0,4000)
    plt.xlabel('mean scores')
    plt.ylabel('# of reviews')
    plt.title('Mean score of papers')
    plt.savefig(os.path.join(output_folder, "o-mean-scores-per-paper.png"))

    # Percent of reviews that are assigned to suggested reviewer
    reviewer_ranks = {}
    for j in range(0,assignment_matrix.shape[1]):
        if paper_mapping[str(j)] in reviewer_suggestions:
            reviewer_assignments = np.where(assignment_matrix[:,j] == 1)[0]
            for r in reviewer_assignments:
                if reviewer_mapping[str(r)] in reviewer_suggestions[paper_mapping[str(j)]]['Reviewers']:
                    rank = reviewer_suggestions[paper_mapping[str(j)]]['Reviewers'].index(reviewer_mapping[str(r)])
                else:
                    rank = -1
                if rank not in reviewer_ranks:
                    reviewer_ranks[rank] = 0
                reviewer_ranks[rank] += 1

    plt.clf()
    plt.bar(list(reviewer_ranks.keys()), list(reviewer_ranks.values()))
    plt.xlabel('Ranks (-1: unranked)')
    plt.ylabel('# of reviewers')
    plt.title('Assignments of suggested reviewers')
    plt.savefig(os.path.join(output_folder, "o-assignments-of-suggested-reviewers.png"))
    
    # Percent of reviewers that have been assigned their full quota
    full_capacity = [0, 0]
    for i,_ in enumerate(reviews_per_reviewer):
        if capacity_vector[i] == reviews_per_reviewer[i]:
            full_capacity[1] += 1
        else:
            full_capacity[0] += 1
    plt.clf()
    plt.bar([0,1], full_capacity)
    plt.xlabel('Full capacity? (0: not full; 1: full)')
    plt.ylabel('# of reviewers')
    plt.title('Capacity limits of reviewers')
    plt.savefig(os.path.join(output_folder, "o-capacity-limits-of-reviewers.png"))
    print ('\tNumber of reviewers at full capacity: {} / {}: {} %'.format( \
        full_capacity[1], full_capacity[0] + full_capacity[1], round(100.0 * full_capacity[1] / (full_capacity[0] + full_capacity[1]), 2)
    ))

    non_suggested_reviewer_list = []

    # Percentage of assigned 1st/2nd/3rd reviewers that are suggested
    plt.clf()
    for n in range(0,num_reviews):
        reviewer_ranks = {}
        plt.subplot(1, 3, n + 1)
        for j in range(0,p_assignment_matrix[n].shape[1]):
            if paper_mapping[str(j)] in reviewer_suggestions:
                reviewer_assignments = np.where(p_assignment_matrix[n][:,j] == 1)[0]
                for r in reviewer_assignments:
                    if reviewer_mapping[str(r)] in reviewer_suggestions[paper_mapping[str(j)]]['Reviewers']:
                        rank = reviewer_suggestions[paper_mapping[str(j)]]['Reviewers'].index(reviewer_mapping[str(r)])
                    else:
                        # record list of non-suggested reviewer/paper pairs
                        non_suggested_reviewer_list.append([paper_mapping[str(j)], reviewer_mapping[str(r)], reviewer_suggestions[paper_mapping[str(j)]]['MetaReviewer']])
                        if n == 0 or n == 1:
                            print ('\t\tN: {}\t\tReviewer: {}(r:{})\t\tPaper: {}(c:{})'.format(n, reviewer_mapping[str(r)], r, paper_mapping[str(j)], j))
                            if os.path.exists(os.path.join(output_folder, 'v-capacities-step-{}.json'.format(0))) and \
                                os.path.exists(os.path.join(output_folder, 'v-capacities-step-{}.json'.format(n))) and \
                                os.path.exists(os.path.join(output_folder, 'v-capacities-step-{}.json'.format(n+1))):

                                capacities_initial = load(os.path.join(output_folder, 'v-capacities-step-{}.json'.format(0)))
                                capacities = load(os.path.join(output_folder, 'v-capacities-step-{}.json'.format(n)))
                                capacities_next_step = load(os.path.join(output_folder, 'v-capacities-step-{}.json'.format(n+1)))

                                for rr, rs in enumerate(reviewer_suggestions[paper_mapping[str(j)]]['Reviewers']):
                                    print ('\t\t\tReviewer: {} (rank: {})\t\t Capacity(initial, prior, after): {}, {}, {}'.format(\
                                        rs, rr, capacities_initial[rs], capacities[rs], capacities_next_step[rs]
                                        ))

                        rank = -1
                    if rank not in reviewer_ranks:
                        reviewer_ranks[rank] = 0
                    reviewer_ranks[rank] += 1
            elif n == 0:
                r = paper_mapping[str(j)]
                print ('\t\tNo Suggestions for Paper: {}\t\tN: {}\t\tReviewer: {}\t\tPaper: {}'.format(paper_mapping[str(j)], \
                        n, reviewer_mapping[str(r)], paper_mapping[str(j)]))

        rank_total_sum = 0.0
        rank_total_population = 0.0
        flattened_ranks = []
        for k in reviewer_ranks.keys():
            rank = int(k)
            if rank > -1:
                rank_total_sum += reviewer_ranks[k] * (rank + 1)
                rank_total_population += reviewer_ranks[k]
                flattened_ranks.extend([rank + 1]*reviewer_ranks[k])

        
        plt.xlim(-1, 25)
        plt.ylim(0, 6000)
        plt.bar(list(reviewer_ranks.keys()), list(reviewer_ranks.values()))
        plt.xlabel('Ranks (-1: unranked)')
        plt.ylabel('# of reviewers')
        med = statistics.median(flattened_ranks)
        avg = round(statistics.mean(flattened_ranks),2)
        if n == 0:
            plt.title('1st reviewer (mean: {} median: {})'.format(avg, med))
        elif n == 1:
            plt.title('2nd reviewer (mean: {} median: {})'.format(avg, med))
        elif n == 2:
            plt.title('3rd reviewer (mean: {} median: {})'.format(avg, med))

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(output_folder, "o-assignments-of-suggested-reviewers-detailed.png"))

    plt.clf()
    non_suggested_tpms_scores = []
    non_suggested_subject_area_scores = []
    non_suggested_suggestion_scores = []
    non_suggested_experience_scores = []
    non_suggested_reviewer_mask = np
    for j in range(0,assignment_matrix.shape[1]):
        for i in range(0,assignment_matrix.shape[0]):
            if assignment_matrix[i,j] < 1.0:
                continue
            if paper_mapping[str(j)] in reviewer_suggestions:
                if reviewer_mapping[str(i)] not in reviewer_suggestions[paper_mapping[str(j)]]['Reviewers']:
                    # if (suggestion_scores_matrix[i,j]) > 0.5:

                    non_suggested_tpms_scores.append(tpms_scores_matrix[i,j])
                    non_suggested_subject_area_scores.append(subject_area_scores_matrix[i,j])
                    non_suggested_suggestion_scores.append(suggestion_scores_matrix[i,j])
            else:
                non_suggested_tpms_scores.append(tpms_scores_matrix[i,j])
                non_suggested_subject_area_scores.append(subject_area_scores_matrix[i,j])
                non_suggested_suggestion_scores.append(suggestion_scores_matrix[i,j])

        
    plt.subplot(3, 1, 1)
    
    plt.xlim(0, 2.0)
    plt.ylim(0, 2500)
    plt.hist(np.array(non_suggested_tpms_scores), bins=25, range=(0,2.0))
    plt.title('TPMS scores for non suggested reviewers (Mean: {} Median:{}'.format(
        round(statistics.mean(non_suggested_tpms_scores),2), statistics.median(non_suggested_tpms_scores)
        ))
    
    plt.subplot(3, 1, 2)
    
    plt.xlim(0, 2.0)
    plt.ylim(0, 2500)
    plt.hist(np.array(non_suggested_subject_area_scores), bins=25, range=(0,2.0))
    plt.title('Subject area scores for non suggested reviewers (Mean: {} Median:{}'.format(
        round(statistics.mean(non_suggested_subject_area_scores),2), statistics.median(non_suggested_subject_area_scores)
        ))
    
    plt.subplot(3, 1, 3)

    plt.xlim(0, 2.0)
    plt.ylim(0, 2500)
    plt.hist(np.array(non_suggested_suggestion_scores), bins=25, range=(0,2.0))
    plt.title('Suggestion scores for non suggested reviewers (Mean: {} Median:{}'.format(
        round(statistics.mean(non_suggested_suggestion_scores),2), statistics.median(non_suggested_suggestion_scores)
        ))


    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.savefig(os.path.join(output_folder, "o-mean-scores-non-suggested-reviewers-detailed.png"))

def output_capacities(capacity_vector, reviewer_mapping, label, output_folder):
    capacities = {}
    for i,_ in enumerate(capacity_vector):
        capacities[reviewer_mapping[str(i)]] = str(capacity_vector[i])
    save(capacities, os.path.join(output_folder, 'v-capacities-{}.json'.format(label)))

def assign_papers(scores_matrix, conflicts_matrix, capacity_vector, num_reviews, num_greedy_steps, reviewer_mapping, output_folder='./', debug=False):
    use_or = True
    print ('*'*50 + ' Assigning papers ' + '*'*50)
    p_assignment_matrix = [ None ] * num_reviews
    num_reviews_ = num_reviews - num_greedy_steps
    conflicts_matrix_ = copy.deepcopy(conflicts_matrix)
    capacity_vector_ = copy.deepcopy(capacity_vector)
    scores_matrix_ = copy.deepcopy(scores_matrix)
    scores_matrix_[scores_matrix_ == 0.0] = 0.000001
    for g in range(0, num_greedy_steps):
        print ('\tSolving min cost flow problem using greedy approach (min cost flow problem with 1 review/paper) - {}/{}'.format(g+1, num_greedy_steps))
        output_capacities(capacity_vector_, reviewer_mapping, 'step-{}'.format(g), output_folder)
        p_assignment_matrix[g], conflicts_matrix_, capacity_vector_ = solve_assigment_problem_greedy(scores_matrix, conflicts_matrix_, capacity_vector_, debug=debug)
    output_capacities(capacity_vector_, reviewer_mapping, 'step-{}'.format(num_greedy_steps), output_folder)

    if num_reviews_ > 0:
        if use_or:
            print ('\tSolving min cost flow problem using OR-Tools - number of reviews needed: {}'.format(num_reviews_))
            assignment_matrix_ = solve_assigment_problem_or_tools(scores_matrix, conflicts_matrix_, capacity_vector_, num_reviews_, debug=debug)
        else:
            print ('\tSolving min cost flow problem using networkx')
            assignment_matrix_ = solve_assigment_problem_networkx(scores_matrix, conflicts_matrix_, capacity_vector_, num_reviews_, debug=debug)
        
        max_scores = [ None ] * num_reviews_

        for i in range(0, num_reviews_):
            max_scores[i] = np.argmax(assignment_matrix_.astype(np.float) * scores_matrix_.astype(np.float), axis=0)
            p_assignment_matrix[i + num_greedy_steps] = np.zeros(scores_matrix_.shape).astype(np.float)
            for j, m_i in enumerate(max_scores[i]):
                scores_matrix_[m_i, j] = 0.0
                p_assignment_matrix[i + num_greedy_steps][m_i, j] = 1
    
    assignment_matrix = np.zeros(scores_matrix.shape).astype(np.float)
    for k in range(0, num_reviews):
        assignment_matrix += p_assignment_matrix[k]

    return p_assignment_matrix, assignment_matrix.astype(np.float)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except os.error as exc:
        pass

def output(num_reviews, cached_folder, output_folder):
    print ('*'*50 + ' Outputting data ' + '*'*50)

    reviewers = load(os.path.join(cached_folder, 'reviewers.json'))
    papers = load(os.path.join(cached_folder, 'papers.json'))
    users = load(os.path.join(cached_folder, 'users.json'))
    reviewer_mapping = load(os.path.join(cached_folder, 'reviewer-mapping.json'))
    paper_mapping = load(os.path.join(cached_folder, 'paper-mapping.json'))
    
    scores_matrix = np.load(os.path.join(cached_folder, 'scores.npy'))
    conflicts_matrix = np.load(os.path.join(cached_folder, 'conflicts.npy'))
    capacity_vector = np.load(os.path.join(cached_folder, 'capacities.npy'))

    # List of M papers: one paper id per line
    with open(os.path.join(output_folder, 'o-papers.csv'), 'w') as fout:
        for p in sorted(list(map(str, papers.keys()))):
            fout.write('{}\n'.format(p))

    # List of N reviewers: one reviewer id per line  (reviewer id is an email address)
    with open(os.path.join(output_folder, 'o-reviewers.csv'), 'w') as fout:
        for r in sorted(reviewers.keys()):
            fout.write('{}\n'.format(r))

    # Conflict info: per line - [paper id, reviewer id] for each pair that has a conflict
    with open(os.path.join(output_folder, 'o-conflicts.csv'), 'w') as fout:
        for i in range(0,conflicts_matrix.shape[0]):
            for j in range(0,conflicts_matrix.shape[1]):
                if conflicts_matrix[i,j] == 1:
                    fout.write('{}, {}\n'.format(paper_mapping[str(j)], reviewer_mapping[str(i)]))

    # Matching scores: per line - [paper id, reviewer id, score]
    with open(os.path.join(output_folder, 'o-matching-scores.csv'), 'w') as fout:
        for i in range(0,scores_matrix.shape[0]):
            for j in range(0,scores_matrix.shape[1]):
                fout.write('{}, {}, {}\n'.format(paper_mapping[str(j)], reviewer_mapping[str(i)], scores_matrix[i,j]))
    
    # Max papers per reviewer: per line - [reviewer id, max #]
    with open(os.path.join(output_folder, 'o-max-reviewers-per-paper.csv'), 'w') as fout:
        for i in range(0,len(capacity_vector)):
            fout.write('{}, {}\n'.format(reviewer_mapping[str(i)], capacity_vector[i]))

    # Reviewers per paper: per line - [reviewer id, min #, max #] (note that min=max in our case)
    with open(os.path.join(output_folder, 'o-reviewers-per-paper.csv'), 'w') as fout:
        for r in sorted(reviewers.keys()):
            fout.write('{}, {}, {}\n'.format(r, num_reviews, num_reviews))

def remove(p):
    if os.path.isdir(p):
        shutil.rmtree(p)
    else:
        if os.path.exists(p):
            os.remove(p)

def load_xml_assignments(fn, reverse_paper_mapping, reverse_reviewer_mapping, assignment_shape):
    assignment_matrix = np.zeros(assignment_shape)
    assignments = minidom.parse(fn)
    for s in assignments.getElementsByTagName('submission'):
        paperid = s.attributes['submissionId'].value
        for u in s.getElementsByTagName('user'):
            reviewerid = u.attributes['email'].value
            assignment_matrix[int(reverse_reviewer_mapping[reviewerid]), int(reverse_paper_mapping[paperid])] = 1
    return assignment_matrix

def save_reviewer_capacities_file(reviewer_adjusted_capacities, fn):
    with open(fn, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Email', 'Computed Capacity'])
        for pair in reviewer_adjusted_capacities:
            writer.writerow(pair)

def main():
    parser = ArgumentParser(description='')
    parser.add_argument('-u', '--users_txt', help='Users.txt as exported from CMT (contains conflict information)')
    parser.add_argument('-r', '--reviewers_csv', help='reviewers.csv as copied and pasted from CMT')
    parser.add_argument('-c', '--conflicts_txt', help='ReviewerConflicts.txt as exported from CMT')
    parser.add_argument('-t', '--tpms_csv', help='ReviewerTpmsScores_CVPR2019 csv file that contains TPMS scores')
    parser.add_argument('-p', '--papers_csv', help='Papers.xls as exported from CMT (removed top 3 empty rows and converted xls file to csv)')
    parser.add_argument('-s', '--reviewer_suggestions_txt', help='ReviewerSuggestions.txt as exported from CMT')
    parser.add_argument('-i', '--cached_folder', default='', help='Cached folder with results')
    parser.add_argument('-n', '--num_reviews', help='Number of reviews per paper')
    parser.add_argument('-g', '--num_greedy_steps', help='Number of greedy steps to run (min: 0, max: num_reviews)')
    parser.add_argument('-w_t', '--w_t', default=1.0, help='Weight for TPMS scores')
    parser.add_argument('-w_a', '--w_a', default=1.0, help='Weight for subject area scores')
    parser.add_argument('-w_s', '--w_s', default=1.0, help='Weight for reviewer suggestions')
    parser.add_argument('-w_e', '--w_e', default=1.0, help='Weight for reviewer experience')
    parser.add_argument('-x', '--cmt_assignments', help='CMT Assignments file (XML format)')
    parser.add_argument('-o', '--config', help='Configuration for different user type quotas')
    parser.add_argument('--debug', dest='debug', action='store_true', help='Debug (verbose mode)')
    
    parser.set_defaults(debug=False)
    parser_options = parser.parse_args()

    start = timer()
    output_folder = os.path.join('output-w_t-{}-w_a-{}-w_s-{}-w_e-{}-g-{}-n-{}-config-{}'.format(\
        parser_options.w_t, parser_options.w_a, parser_options.w_s, parser_options.w_e, \
        parser_options.num_greedy_steps, parser_options.num_reviews, parser_options.config
    ))

    if str.lower(parser_options.config) == 'high':
        reviewer_capacities = {
            'Researcher/faculty': 10,
            'Student': 6,
            '': 6
        }
    elif str.lower(parser_options.config) == 'low':
        reviewer_capacities = {
            'Researcher/faculty': 9,
            'Student': 5,
            '': 5
        }


    mkdir_p(output_folder)
    
    if parser_options.cmt_assignments:
        assignment_matrix_ours = np.load(os.path.join(parser_options.cached_folder, 'assignments.npy'))

        reviewer_mapping = load(os.path.join(parser_options.cached_folder, 'reviewer-mapping.json'))
        paper_mapping = load(os.path.join(parser_options.cached_folder, 'paper-mapping.json'))
        reverse_reviewer_mapping = load(os.path.join(parser_options.cached_folder, 'reverse-reviewer-mapping.json'))
        reverse_paper_mapping = load(os.path.join(parser_options.cached_folder, 'reverse-paper-mapping.json'))
        reviewer_suggestions = load(os.path.join(parser_options.cached_folder, 'reviewer_suggestions.json'))
        scores_matrix = np.load(os.path.join(parser_options.cached_folder,'scores.npy'))
        tpms_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'tpms_scores.npy'))
        subject_area_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'subject_area_scores.npy'))
        suggestion_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'suggestion_scores.npy'))
        experience_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'experience_scores.npy'))
        conflicts_matrix = np.load(os.path.join(parser_options.cached_folder,'conflicts.npy'))
        capacity_vector = np.load(os.path.join(parser_options.cached_folder, 'capacities.npy'))
        
        assignment_matrix = load_xml_assignments(parser_options.cmt_assignments, reverse_paper_mapping, reverse_reviewer_mapping, assignment_matrix_ours.shape)

        num_reviews = int(parser_options.num_reviews)
        num_reviews_ = int(parser_options.num_reviews)
        num_greedy_steps = 0
        p_assignment_matrix = [ None ] * num_reviews
        scores_matrix_ = copy.deepcopy(scores_matrix)
        assignment_matrix_ = copy.deepcopy(assignment_matrix)
        scores_matrix_[scores_matrix_ == 0.0] = 0.000001
        max_scores = [ None ] * num_reviews_
        for i in range(0, num_reviews_):
            max_scores[i] = np.argmax(assignment_matrix_.astype(np.float) * scores_matrix_.astype(np.float), axis=0)
            print ('{}'.format(i + num_greedy_steps))
            p_assignment_matrix[i + num_greedy_steps] = np.zeros(scores_matrix_.shape).astype(np.float)
            for j, m_i in enumerate(max_scores[i]):
                scores_matrix_[m_i, j] = 0.0
                p_assignment_matrix[i + num_greedy_steps][m_i, j] = 1

        output_folder = 'external-assignments'
        mkdir_p(output_folder)
        validate_results(output_folder, reviewer_suggestions, p_assignment_matrix, assignment_matrix, scores_matrix, \
            tpms_scores_matrix, subject_area_scores_matrix, suggestion_scores_matrix, experience_scores_matrix, conflicts_matrix, capacity_vector, \
            reviewer_mapping, paper_mapping, reverse_reviewer_mapping, int(parser_options.num_reviews))
        return

    if not os.path.exists(os.path.join(parser_options.cached_folder, 'scores.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'tpms_scores.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'subject_area_scores.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'suggestion_scores.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'experience_scores.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'conflicts.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'capacities.npy')):

        if os.path.exists(os.path.join(parser_options.cached_folder, 'reviewers.json')) and \
            os.path.exists(os.path.join(parser_options.cached_folder, 'papers.json')) and \
            os.path.exists(os.path.join(parser_options.cached_folder, 'users.json')) and \
            os.path.exists(os.path.join(parser_options.cached_folder, 'paper_conflicts.json')) and \
            os.path.exists(os.path.join(parser_options.cached_folder, 'reviewer_suggestions.json')):

            print ('Loading cached reviewers and papers files')
            reviewers = load(os.path.join(parser_options.cached_folder, 'reviewers.json'))
            papers = load(os.path.join(parser_options.cached_folder, 'papers.json'))
            reviewer_suggestions = load(os.path.join(parser_options.cached_folder, 'reviewer_suggestions.json'))
            users = load(os.path.join(parser_options.cached_folder, 'users.json'))
            paper_conflicts = load(os.path.join(parser_options.cached_folder, 'paper_conflicts.json'))
        else:
            print ('Calculating final scores and capacities...')
            papers = parse_papers(parser_options.papers_csv)
            users, organizations = parse_users(parser_options.users_txt)
            paper_conflicts = parse_conflicts(parser_options.conflicts_txt)
            reviewers, domains = parse_reviewers(parser_options.reviewers_csv, users, organizations)
            reviewer_suggestions = parse_reviewer_suggestions(parser_options.reviewer_suggestions_txt)

            paper_ids = set(papers.keys())
            paper_ids_suggestions = set(reviewer_suggestions.keys())
            print('Number of papers without reviewer suggestions:', len(paper_ids-paper_ids_suggestions))

            parse_and_calculate_scores(parser_options.tpms_csv, reviewers, papers, reviewer_suggestions, \
                float(parser_options.w_t), float(parser_options.w_a), float(parser_options.w_s), float(parser_options.w_e), debug=parser_options.debug)
            
            print ('\tSaving intermediate reviewers and papers files')
            save(reviewers, os.path.join(output_folder, 'reviewers.json'))
            save(paper_conflicts, os.path.join(output_folder, 'paper_conflicts.json'))
            save(papers, os.path.join(output_folder, 'papers.json'))
            save(users, os.path.join(output_folder, 'users.json'))
            save(reviewer_suggestions, os.path.join(output_folder, 'reviewer_suggestions.json'))

        print('Calculating scores/capacities')
        scores_matrix, tpms_scores_matrix, subject_area_scores_matrix, suggestion_scores_matrix, experience_scores_matrix, \
            conflicts_matrix, capacity_vector, reviewer_adjusted_capacities, reviewer_mapping, paper_mapping, reverse_reviewer_mapping, reverse_paper_mapping = \
            formulate_matrices(reviewers, reviewer_capacities, paper_conflicts, papers, users, debug=parser_options.debug)
        np.save(os.path.join(output_folder, 'scores.npy'), scores_matrix)
        np.save(os.path.join(output_folder, 'tpms_scores.npy'), tpms_scores_matrix)
        np.save(os.path.join(output_folder, 'subject_area_scores.npy'), subject_area_scores_matrix)
        np.save(os.path.join(output_folder, 'suggestion_scores.npy'), suggestion_scores_matrix)
        np.save(os.path.join(output_folder, 'experience_scores.npy'), experience_scores_matrix)
        np.save(os.path.join(output_folder, 'conflicts.npy'), conflicts_matrix)
        np.save(os.path.join(output_folder, 'capacities.npy'), capacity_vector)
        save_reviewer_capacities_file(reviewer_adjusted_capacities, os.path.join(output_folder, 'reviewer-capacities.csv'))
        save(reviewer_mapping, os.path.join(output_folder, 'reviewer-mapping.json'))
        save(paper_mapping, os.path.join(output_folder, 'paper-mapping.json'))
        save(reverse_reviewer_mapping, os.path.join(output_folder, 'reverse-reviewer-mapping.json'))
        save(reverse_paper_mapping, os.path.join(output_folder, 'reverse-paper-mapping.json'))
    else:
        print ('Final scores and capacities already calculated - loading cached files...')
        scores_matrix = np.load(os.path.join(parser_options.cached_folder,'scores.npy'))
        tpms_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'tpms_scores.npy'))
        subject_area_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'subject_area_scores.npy'))
        suggestion_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'suggestion_scores.npy'))
        experience_scores_matrix = np.load(os.path.join(parser_options.cached_folder,'experience_scores.npy'))
        conflicts_matrix = np.load(os.path.join(parser_options.cached_folder,'conflicts.npy'))
        capacity_vector = np.load(os.path.join(parser_options.cached_folder, 'capacities.npy'))
        reviewer_mapping = load(os.path.join(parser_options.cached_folder, 'reviewer-mapping.json'))
        paper_mapping = load(os.path.join(parser_options.cached_folder, 'paper-mapping.json'))
        reverse_reviewer_mapping = load(os.path.join(parser_options.cached_folder, 'reverse-reviewer-mapping.json'))
        reverse_paper_mapping = load(os.path.join(parser_options.cached_folder, 'reverse-paper-mapping.json'))
        reviewers = load(os.path.join(parser_options.cached_folder, 'reviewers.json'))
        papers = load(os.path.join(parser_options.cached_folder, 'papers.json'))
        reviewer_suggestions = load(os.path.join(parser_options.cached_folder, 'reviewer_suggestions.json'))
        users = load(os.path.join(parser_options.cached_folder, 'users.json'))

    if not os.path.exists(os.path.join(parser_options.cached_folder, 'assignments.npy')) or \
        not os.path.exists(os.path.join(parser_options.cached_folder, 'assignments.xml')):
        print ('Calculating assignments...')
        p_assignment_matrix, assignment_matrix = assign_papers(scores_matrix, conflicts_matrix, capacity_vector, int(parser_options.num_reviews), \
            int(parser_options.num_greedy_steps), reviewer_mapping, output_folder=output_folder, debug=parser_options.debug)

        for ii in range(0,int(parser_options.num_reviews)):
            np.save(os.path.join(output_folder, 'assignments-R{}.npy'.format(ii)), p_assignment_matrix[ii])
        np.save(os.path.join(output_folder, 'assignments.npy'), assignment_matrix)
        save_xml_assignments(output_folder, assignment_matrix, reviewer_mapping, paper_mapping)
    else:
        print ('Assignments already calculated - loading cached files...')
        assignment_matrix = np.load(os.path.join(parser_options.cached_folder, 'assignments.npy'))
        p_assignment_matrix = [None] * int(parser_options.num_reviews)
        for ii in range(0,int(parser_options.num_reviews)):
            p_assignment_matrix[ii] = np.load(os.path.join(parser_options.cached_folder, 'assignments-R{}.npy'.format(ii)))



    assignment_matrix = load_xml_assignments(os.path.join(output_folder, 'assignments.xml'), reverse_paper_mapping, reverse_reviewer_mapping, assignment_matrix.shape)

    validate_results(output_folder, reviewer_suggestions, p_assignment_matrix, assignment_matrix, scores_matrix, \
        tpms_scores_matrix, subject_area_scores_matrix, suggestion_scores_matrix, experience_scores_matrix, conflicts_matrix, capacity_vector, \
        reviewer_mapping, paper_mapping, reverse_reviewer_mapping, int(parser_options.num_reviews))

    output(int(parser_options.num_reviews), parser_options.cached_folder, output_folder)

    end = timer()
    print ('Total time: {}'.format(end-start))

if __name__ == '__main__':
    main()
