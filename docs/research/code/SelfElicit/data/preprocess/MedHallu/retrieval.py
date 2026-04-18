import os
import json
import requests
from tqdm import tqdm
from split_utils import write


def online_kuake_search(query, type='bing'):
    assert type in ['quark', 'bing']
    cookie = "sm_uuid=db51a54090544ed4f9253462f36c7acd%7C%7C%7C1698460804; sm_diu=db51a54090544ed4f9253462f36c7acd%7C%7C1Fe0ffe07a236b9b4e%7C1698460804; __itrace_wid=6dce52ae-3a7e-488f-2b16-2d971e3454b7; sm_ruid=1f4e27e8b37f98f31fea20aca48d0f77%7C%7C%7C1698655868; cna=dPybHd1TmWcBASQBsYCtpg57; XSRF-TOKEN=910a958e-579d-4263-b9ef-3af12fd03bda; sm_sid=e80e6621862ab6b954600a8f37b3119d; lsmap2=3M64U07S1CE1Dq0Nn0Ps0Qj0Tx0WkTYw0ht0hu0oi1oj1sw4sx3; phid=e80e6621862ab6b954600a8f37b3119d; x5sec=7b22733b32223a2266616531316661333434393039353638222c2277616762726964676561643b32223a223138373639373962363236306539323266323262333039666439336636323237434b503673616f47454b53596e4a7a362f2f2f2f2f77457736376174692f2f2f2f2f2f2f41554144227d; isg=BISEcPiHvTtnmAmHOoRWanI7VQR2nagHYfOtJ54lsc8SySWTxq8PlbZoDGERV-Bf; l=fBNqV3y4PII9eNTdBO5Inurza779NIRb8rVzaNbMiIEGa6gfTFOuZNCT9qsJ7dtxgTfc2etzz7IVtdh95lz_8dkDBeYIb5GRqqJ9-bpU-L5..; tfstk=dRpX08O46FprIdCT3xiyP6EhYGW6CEMekls9xhe4XtBABRLwbt5OunR55U_lk-WxnU8ywQ2NgVDDzURJZZQN0hk1WEXAsI7NuN66jeir8vkemnXG32uELESYD9DJC9b9mG-cIO3U4RhEdnYeej-AURDbRojj6wy63uq_F6N1KRe9hAffRS_RqgJ5Ms_dga6ZBJ7I3yN_V621VwoSV5VGsTgJ3"
    url = "http://eas-zhangbei-b.alibaba-inc.com/api/predict/online_kuake_search"
    data = {"query": query, "cookie": cookie, "type": type}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.84 Safari/537.36",
        "Authorization": "MjAzYWZlODg1OWViZDFkYmM2YjQ3ODNiNTZmMTNhOWU5OTAwNDdkMA=="
    }
    retrievals = None
    while not isinstance(retrievals, list):
        response = requests.get(url, headers=headers, json=data)
        response.raise_for_status()

        search_res = json.loads(response.text)
        search_res = search_res['bing_all']
        bing_res = json.loads(search_res)
        retrievals = bing_res['webPages']['value']

    return retrievals


if __name__ == '__main__':
    path = "./data/cache/MedHallu/export"
    files = list(filter(lambda file: os.path.splitext(file)[-1] == '.jsonl', os.listdir(path)))
    files = sorted(files)

    all_samples = {}
    for file in files:
        print(f"Loading datasets from {file}")

        samples = []
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                sample = json.loads(line)
                assert all([k in sample for k in
                            ['question', 'answer', 'claims', 'sentences', 'label', 'from', 'claim_labels',
                             'sentence_labels']])
                samples.append(sample)

        file = file.replace('.jsonl', '')
        all_samples[file] = samples

    for file in all_samples:
        os.makedirs(os.path.join("./data/cache/MedHallu/retrieval/", file), exist_ok=True)
        for i, sample in enumerate(all_samples[file]):
            cache_path = os.path.join("./data/cache/MedHallu_zh/retrieval/", file, f"{i}.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding='utf-8') as f:
                        data = json.load(f)
                        claim_retrievals = data["claim_retrievals"]
                        sentence_retrievals = data["sentence_retrievals"]
                except:
                    print(cache_path)
                    continue

            else:
                claim_retrievals = []
                for claim in tqdm(sample['claims'], desc=f"{file}-{i}-claim"):
                    retrieval_result = online_kuake_search(claim)
                    claim_retrievals.append(retrieval_result)

                sentence_retrievals = []
                for sentence in tqdm(sample['sentences'], desc=f"{file}-{i}-sentence"):
                    retrieval_result = online_kuake_search(sentence)
                    sentence_retrievals.append(retrieval_result)

                with open(cache_path, "w", encoding='utf-8') as f:
                    json.dump({'claim_retrievals': claim_retrievals,
                               'sentence_retrievals': sentence_retrievals}, f, ensure_ascii=False)

            assert len(claim_retrievals) == len(sample['claims'])
            if len(sentence_retrievals) != len(sample['sentences']):
                print(cache_path)
                os.remove(cache_path)

            claim_retrievals = [[ele['snippet'].replace('\n', '') for ele in item] for item in claim_retrievals]
            sentence_retrievals = [[ele['snippet'].replace('\n', '') for ele in item] for item in sentence_retrievals]
            sample['claim_retrievals'] = claim_retrievals
            sample['sentence_retrievals'] = sentence_retrievals

    for file in all_samples:
        write(all_samples[file], file, dir="./data/dataset/MedHallu/")
