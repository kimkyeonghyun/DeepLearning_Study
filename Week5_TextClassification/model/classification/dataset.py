# text_classification
# dataset

# 라이브러리
import pickle
import argparse
from tqdm.auto import tqdm
from torch.utils.data.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_path: str) -> None:
        super(CustomDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)
            # 텍스트 이외의 자료혀을 파일로 저장하기 위해 pickle모듈 이용

        self.data_list = []
        self.vocab_size = data_['vocab_size']
        self.num_classes = data_['num_classes']

        for idx in tqdm(range(len(data_['texts'])), desc = f'Loading data from {data_path}'):

            # print(len(data_['texts']),idx)
            self.data_list.append({
                'texts': data_['texts'][idx],
                'labels': data_['labels'][idx],
                'attention_mask': data_['attention_mask'][idx],
                'token_type_ids': data_['token_type_ids'][idx],
                'index': idx
            })

        del data_
    # 클래스의 인덱스에 접근할 때 자동으로 호출되는 메서드 - 해당 인덱스의 데이터를 반환
    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
