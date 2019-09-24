import csv
import glob
import hashlib
import uuid
import os
from argparse import ArgumentParser


hashmap = {}

def encrypt_string(hash_string):
    hash_string = hash_string.strip().replace('"','')
    if not hash_string in hashmap:
        hashmap[hash_string] = uuid.uuid4().hex
    return hashmap[hash_string]


def encrypt_subjectlist(subjectlist_string):
    return ';'.join([encrypt_string(val) for val in subjectlist_string.split(';')])


def obfuscate_reviewers(folder, reviewers_csv):
    with open(os.path.join(folder, 'o-{}'.format(reviewers_csv)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter='\t', quotechar='"')
        with open(os.path.join(folder, reviewers_csv)) as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0:
                    spamwriter.writerow(datum)
                    continue
                
                firstname, lastname, email, org, quota, assigned, comp, comp_percent, bids, conflicts_entered, user_type, ext_prof_entered, sa_selected, sa_primary, sa_secondary, \
                    _ = datum
                spamwriter.writerow([
                    encrypt_string(firstname), encrypt_string(lastname), encrypt_string(email), encrypt_string(org), quota, assigned, comp, comp_percent, bids, \
                    conflicts_entered, user_type, ext_prof_entered, sa_selected, encrypt_subjectlist(sa_primary), encrypt_subjectlist(sa_secondary), _ 
                    ])

def obfuscate_quotas(folder, quotas_fn):
    with open(os.path.join(folder, 'o-{}'.format(quotas_fn)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter='\t', quotechar='"')
        with open(os.path.join(folder, quotas_fn)) as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0 or (d + 1) % 2 == 0:
                    if d == 0:
                        spamwriter.writerow(datum)
                    else:
                        spamwriter.writerow([0])
                    continue

                firstname, lastname, email, assigned, comp, comp_percent, user_type, quota, _ = datum
                spamwriter.writerow([
                    encrypt_string(firstname), encrypt_string(lastname), encrypt_string(email), assigned, comp, comp_percent, user_type, quota, _
                    ])

def obfuscate_papers(folder, papers_fn):
    with open(os.path.join(folder, 'o-{}'.format(papers_fn)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter=',', quotechar='"')
        with open(os.path.join(folder, papers_fn), encoding='latin-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0:
                    spamwriter.writerow(datum)
                    continue

                paper_id, created, modified, title, abstract, author_names, author_emails, track_name, primary_sa, secondary_sa, \
                conflicts, assigned, completed, percent_completed, bids, discussion, status, requested_camera_ready, camera_ready_submitted, \
                requested_author_feedback, author_feedback_submitted, files, num_files, supplementary_files, num_supplementary_files, \
                reviewers, reviewer_emails, metareviewers, metareviewer_emails, sr_metareviewers, sr_metareviewer_emails, q3, q4, q5 = datum
                
                o_author_emails = []
                for i,e in enumerate(author_emails.split(';')):
                    if len(e.split('*')) > 1:
                        o_author_emails.append(encrypt_string(e.split('*')[0]) + '*')
                    else:
                        o_author_emails.append(encrypt_string(e))
                o_author_emails_str = ';'.join(o_author_emails)

                spamwriter.writerow([
                    encrypt_string(paper_id), created, modified, encrypt_string(title), encrypt_string(abstract), encrypt_string(author_names), \
                    o_author_emails_str, track_name, encrypt_subjectlist(primary_sa), encrypt_subjectlist(secondary_sa), \
                    conflicts, assigned, completed, percent_completed, bids, discussion, status, requested_camera_ready, camera_ready_submitted, \
                    requested_author_feedback, author_feedback_submitted, encrypt_string(files), num_files, encrypt_string(supplementary_files), num_supplementary_files, \
                    encrypt_string(reviewers), encrypt_string(reviewer_emails), encrypt_string(metareviewers), encrypt_string(metareviewer_emails), \
                    encrypt_string(sr_metareviewers), encrypt_string(sr_metareviewer_emails), q3, q4, q5
                    ])

def obfuscate_conflicts(folder, conflicts_fn):
    with open(os.path.join(folder, 'o-{}'.format(conflicts_fn)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter='\t', quotechar='"')
        with open(os.path.join(folder, conflicts_fn)) as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0:
                    spamwriter.writerow(datum)
                    continue

                paper_id, reviewer_email = datum
                spamwriter.writerow([
                    encrypt_string(paper_id), encrypt_string(reviewer_email)
                    ])

def obfuscate_reviewer_suggestions(folder, reviewer_suggestions_fn):
    with open(os.path.join(folder, 'o-{}'.format(reviewer_suggestions_fn)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter='\t', quotechar='"')
        with open(os.path.join(folder, reviewer_suggestions_fn), encoding='latin-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0:
                    spamwriter.writerow(datum)
                    continue
                
                paper_id, metareviewer_email, reviewer_email, rank = datum
                spamwriter.writerow([
                    encrypt_string(paper_id), encrypt_string(metareviewer_email), encrypt_string(reviewer_email), rank
                    ])

def obfuscate_tpms_scores(folder, tpms_fn):
    with open(os.path.join(folder, 'o-{}'.format(tpms_fn)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter=',', quotechar='"')
        with open(os.path.join(folder, tpms_fn), encoding='latin-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0:
                    spamwriter.writerow(datum)
                    continue
                
                paper_id, email, tpms = datum
                spamwriter.writerow([
                    encrypt_string(paper_id), encrypt_string(email), tpms
                    ])

def obfuscate_users(folder, users_fn):
    with open(os.path.join(folder, 'o-{}'.format(users_fn)), 'w', newline='') as o_csvfile:
        spamwriter = csv.writer(o_csvfile, delimiter='\t', quotechar='"')
        with open(os.path.join(folder, users_fn), encoding='latin-1') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for d, datum in enumerate(spamreader):
                if d == 0:
                    spamwriter.writerow(datum)
                    continue
                
                firstname, _, lastname, email, org, country, conflicts = datum
                # obfuscate conflicts individually
                o_conflicts = ';'.join([encrypt_string(c) for c in conflicts.split(';')])
                spamwriter.writerow([
                    encrypt_string(firstname), _, encrypt_string(lastname), encrypt_string(email), encrypt_string(org), encrypt_string(country), \
                    o_conflicts
                    ])

def obfuscate_folder(folder):
    for f in glob.glob(folder + '/*'):
        fn = f.split('/')[-1]
        if fn == 'reviewers.csv':
            obfuscate_reviewers(folder, fn)
        elif fn == 'quotas.csv':
            obfuscate_quotas(folder, fn)
        elif fn == 'Papers.csv':
            obfuscate_papers(folder, fn)
        elif fn == 'ReviewerConflicts.txt':
            obfuscate_conflicts(folder, fn)
        elif fn == 'ReviewerSuggestions.txt':
            obfuscate_reviewer_suggestions(folder, fn)
        elif fn == 'ReviewerTpmsScores_ICCV2019.csv':
            obfuscate_tpms_scores(folder, fn)
        elif fn == 'Users.txt':
            obfuscate_users(folder, fn)

def main():
    parser = ArgumentParser(description='')
    parser.add_argument('-d', '--folder', help='Directory where we can find data. Obfuscated data gets put in the same folder.')
    parser_options = parser.parse_args()

    obfuscate_folder(parser_options.folder)

if __name__ == '__main__':
    main()
    