from plot.quality.fact_getdata import *

if __name__ == '__main__':
    all_claims, all_reflections = {}, {}
    all_claims['IO'], all_reflections['IO'] = get_io_log(paths['IO'])
    all_claims['FaR'], all_reflections['FaR'] = get_far_log(paths['FaR'])
    all_claims['context'], all_reflections['context'] = get_io_log(paths['context'])
    all_claims['elicit'], all_reflections['elicit'] = get_elicit_log(paths['elicit'])
    all_claims['cot'], all_reflections['cot'] = get_cot_log(paths['cot'])

    # load model
    model = APIModel(None)

    x = 4
    x = 50 * x
    y = x + 50

    # check rel
    for k in all_reflections:
        file = f"./plot/quality/output/{dataset}/rel-{k}-{x}-{y}.txt"

        # 断点续传
        if not os.path.exists(file):
            with open(file, 'w', encoding='utf-8') as f:
                pass
        with open(file, 'r', encoding='utf-8') as f:
            cur_id = len(f.readlines())
            cur_id = cur_id + x

        loader = tqdm(zip(all_claims[k], all_reflections[k]), total=len(all_claims[k]), desc=k)
        for i, (claim, reflection) in enumerate(loader):
            if not cur_id <= i < y:
                continue

            print(i)

            all_rel = []
            # assert len(claim) == len(reflection)
            for c, r in zip(claim, reflection):
                rels = []
                for item in r:
                    query = f"请判断以下两个句子是否具有相同的意思，并使用**相同**或**不同**来回答。请不要解释。\n"
                    query += f"句子1：{c}\n句子2：{item}"
                    _, response, _ = model.understand(query)
                    loader.set_description(f"{k}-{response}")
                    if '不同' in response or '不相同' in response:
                        rel = 0
                    elif '相同' in response:
                        rel = 1
                    else:
                        rel = 0.5

                    rels.append(rel)

                all_rel.append(rels)

            with open(file, 'a', encoding='utf-8') as f:
                s = json.dumps(all_rel)
                f.write(f'{s}\n')
