def triple_reader(file_path):
    triples = set()
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            subj, prop, obj = line.strip('\n').split('\t')
            triples.add((subj, prop, obj))
    return triples


def pair_reader(file_path):
    pairs = set()
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            i, j = line.strip('\n').split('\t')
            pairs.add((i, j))
    return pairs


def count_linked_triples(triple_path, link_path):
    print(triple_path, link_path)
    triples = triple_reader(triple_path)
    print("total triples:", len(triples))
    links = pair_reader(link_path)
    print("total links:", len(links))
    linked_source_ents = set([link[0] for link in links])
    linked_triples_num = 0
    for h, r, t in triples:
        if h in linked_source_ents or t in linked_source_ents:
            linked_triples_num += 1
    print("linked triples:", linked_triples_num, len(triples), round(linked_triples_num / len(triples), 3))
    print()


if __name__ == '__main__':
    wikidata_small_path = "../dataset/large_kgs/wikidata_small/triples.txt"
    wikidata5m_path = "../dataset/large_kgs/wikidata5m/triples.txt"
    wikidata20m_path = "../dataset/large_kgs/wikidata20m/triples.txt"

    wikidata5m_link_path = "../dataset/large_kgs/wikidata5m/links_fb15k237.txt"
    wikidata_small_link_path = "../dataset/large_kgs/wikidata_small/links_fb15k237.txt"

    # triples1 = triple_reader(wikidata_small_path)
    # triples2 = triple_reader(wikidata5m_path)
    # print(len(triples1), len(triples2), len(triples1 & triples2))
    # triples3 = triple_reader(wikidata20m_path)
    # print(len(triples1), len(triples3), len(triples1 & triples3))

    count_linked_triples(wikidata_small_path, wikidata_small_link_path)
    count_linked_triples(wikidata5m_path, wikidata5m_link_path)
