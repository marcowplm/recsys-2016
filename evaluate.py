def evaluate():
    test_interactions = {}

    for line in open('data_modified/interactions_test.csv', 'r'):
        elems = line.split(',')
        test_interactions[int(elems[0])] = {int(x) for x in elems[1].split()}

    header = False
    count = 0
    bad = 0

    MAP = 0.0
    for line in open('data_modified/result.csv', 'r'):
        if not header:
            header = True
            continue

        count += 1

        elems = line.split(',')
        user_id = int(elems[0])
        if user_id in test_interactions:
            user_test_interactions = test_interactions[user_id]
        else:
            bad += 1
            user_test_interactions = set()
        recommendations = [int(x) for x in elems[1].split()]
        AP = 0.0
        for i in range(1, 6):
            ok = len(user_test_interactions.intersection(recommendations[:i]))
            AP += ok / (i * 5.0)

        MAP += AP

    MAP /= count
    # print(bad)
    print(MAP)


def main():
    evaluate()


if __name__ == "__main__":
    main()
