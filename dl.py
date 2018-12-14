import collections, gzip, json, lxml.html, os, re, urllib.parse, urllib.request

def fetch(cache_key, url, postdata=None):
    path = 'cache/' + cache_key
    try:
        return open(path, 'rb').read()
    except FileNotFoundError:
        pass
    # Need to fill in the headers here.
    if postdata:
        postdata['csrf_token'] = ''
        postdata = urllib.parse.urlencode(postdata).encode('utf-8')
    req = urllib.request.Request(url, postdata)
    req.add_header('User-Agent', '')
    if postdata:
        req.add_header('X-Csrf-Token', '')
        req.add_header('Cookie', '')
    data = urllib.request.urlopen(req).read()
    os.makedirs('cache', exist_ok=True)
    open(path, 'wb').write(data)
    return data

def fetch_contest_standings_page(contest_id, page_num):
    url = 'http://codeforces.com/contest/%s/standings/page/%s' % (contest_id, page_num)
    key = 'contest_standings_%s_%s' % (contest_id, page_num)
    return fetch(key, url)

def fetch_contest_ratings_page(contest_id, page_num):
    url = 'http://codeforces.com/contest/%s/ratings/page/%s' % (contest_id, page_num)
    key = 'contest_ratings_%s_%s' % (contest_id, page_num)
    return fetch(key, url)

def fetch_submission(submission_id):
    url = 'http://codeforces.com/data/submitSource'
    key = 'source_%s' % submission_id
    postdata = {'submissionId': submission_id}
    return fetch(key, url, postdata)

def fetch_all_ratings(contest_id):
    rating_dict = {}
    cur_page = 1
    while True:
        ratings = lxml.html.fromstring(fetch_contest_ratings_page(contest_id, cur_page))
        table = ratings.cssselect('table')[0]
        for tr in table.cssselect('tr'):
            td_links = tr.cssselect('td a')
            if not td_links: continue
            username = re.match(r'/profile/(.*)', td_links[0].get('href')).group(1)
            old_rank = td_links[0].get('title').rsplit(maxsplit=1)[0]
            old_rating = int(tr.cssselect('td')[4].cssselect('span')[0].text)
            new_rating = int(tr.cssselect('td')[4].cssselect('span')[1].text)
            rating_dict[username] = {'old_rank': old_rank, 'old_rating': old_rating}
        is_last_page = 'active' in list(ratings.cssselect('.custom-links-pagination span')[-1].classes)
        if is_last_page: break
        cur_page += 1
    return rating_dict

def fetch_all_solutions(contest_id, rating_dict, jsonout):
    cur_index = 0
    cur_page = 1
    summary = collections.defaultdict(int)
    while True:
        standings = lxml.html.fromstring(fetch_contest_standings_page(contest_id, cur_page))
        table = standings.cssselect('table.standings')[0]
        for tr in table.cssselect('tr'):
            participantId = tr.get('participantid')
            if not participantId: continue
            cur_index += 1
            country_elems = tr.cssselect('img.standings-flag')
            if not country_elems: continue
            country = re.match(r'.*/([a-z]{2})\.png', country_elems[0].get('src')).group(1)
            username = re.match(r'/profile/(.*)', tr.cssselect('a')[0].get('href')).group(1)
            rating = rating_dict[username]
            if rating['old_rank'] == 'Unrated,': continue
            #print(country, username)
            for prob_index, td in enumerate(tr.cssselect('td')):
                submissionId = td.get('acceptedsubmissionid')
                if not submissionId: continue # No accepted submission.
                problemId = td.get('problemid')
                title = td.get('title')
                summary['C++' if 'C++' in title else 'Other'] += 1
                if 'C++' not in title: continue
                #print(problemId, submissionId, title)
                submission = json.loads(fetch_submission(submissionId).decode('utf-8'))
                obj = {
                    'contest': contest_id,
                    'country': country,
                    'problem': chr(ord('A')+prob_index),
                    'source': submission['source'],
                    **rating,
                }
                json.dump(obj, jsonout)
                jsonout.write('\n')
        is_last_page = 'active' in list(standings.cssselect('.custom-links-pagination span')[-1].classes)
        if is_last_page: break
        cur_page += 1
    print('Contest %d: %d total' % (contest_id, sum(summary.values())))
    print(summary)

CONTESTS = [
    1056, # Mail.Ru Cup 2018 Round 3
    1055, # Mail.Ru Cup 2018 Round 2
    1043, # Codeforces Round #519 by Botan Investments
    1054, # Mail.Ru Cup 2018 Round 1
    1033, # Lyft Level 5 Challenge 2018 - Elimination Round
    1060, # Codeforces Round #513 by Barcelona Bootcamp (rated, Div. 1 + Div. 2)
    1037, # Manthan, Codefest 18 (rated, Div. 1 + Div. 2)
    1028, # AIM Tech Round 5 (rated, Div. 1 + Div. 2)
    1025, # Codeforces Round #505 (rated, Div. 1 + Div. 2, based on VK Cup 2018 Final)
    1023, # Codeforces Round #504 (rated, Div. 1 + Div. 2, based on VK Cup 2018 Final)
]
#1032, # Technocup 2019 - Elimination Round 3 (unrated)

with gzip.open('cs229_project/data2.json.gz', 'wt') as jsonout:
    for contest_id in CONTESTS:
        print('Contest', contest_id)
        rating_dict = fetch_all_ratings(contest_id)
        fetch_all_solutions(contest_id, rating_dict, jsonout)
