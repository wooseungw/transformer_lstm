import torch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    어텐션 점수, 가중치, 그리고 출력을 계산합니다.
    
    Args:
    - query (torch.Tensor): 쿼리 텐서, 크기는 (batch_size, num_queries, d_k).
    - key (torch.Tensor): 키 텐서, 크기는 (batch_size, num_keys, d_k).
    - value (torch.Tensor): 값 텐서, 크기는 (batch_size, num_keys, d_v).
    - mask (torch.Tensor, optional): 마스크 텐서, 크기는 (batch_size, num_queries, num_keys).
    
    Returns:
    - output (torch.Tensor): 출력 텐서, 크기는 (batch_size, num_queries, d_v).
    - attention_weights (torch.Tensor): 어텐션 가중치, 크기는 (batch_size, num_queries, num_keys).
    """
    
    # 1. 어텐션 점수 계산
    # 쿼리와 키 간의 내적을 계산합니다.
    attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (batch_size, num_queries, num_keys)
    
    # 어텐션 점수를 스케일링합니다.
    d_k = key.size(-1)
    attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # 제공된 경우 마스크를 적용합니다.
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    
    # 2. 어텐션 가중치 계산
    # 어텐션 점수에 소프트맥스 함수를 적용하여 어텐션 가중치를 얻습니다.
    attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_queries, num_keys)
    
    # 3. 출력 계산
    # 값의 가중 평균을 계산하여 출력을 얻습니다.
    output = torch.matmul(attention_weights, value)  # (batch_size, num_queries, d_v) , 행렬의 곱셈 = 행렬의 내적
    
    return output, attention_weights

# 예제 사용:
batch_size = 32
num_queries = 10
num_keys = 100
d_k = 64
d_v = 32

query = torch.randn(batch_size, num_queries, d_k)
print("Shape of Q:",query.shape)

#키와 쿼리의 개수는 동일,벨류의 개수는 다름
#Q,K,V 는 모두 같은 인풋(임베딩 됨)에서 나옴, but가중치 행렬이 적용되어 사이즈가 달라짐
key = torch.randn(batch_size, num_keys, d_k)
print("Shape of K:",key.shape)
value = torch.randn(batch_size, num_keys, d_v)
print("Shape of V:",value.shape)

output, attention_weights = scaled_dot_product_attention(query, key, value)
print(output.shape)  # 예상: torch.Size([32, 10, 64])
print(attention_weights.size)  # 예상: torch.Size([32, 10, 100])
print((attention_weights[0][:][:]))  