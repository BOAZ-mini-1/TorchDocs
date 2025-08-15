import json
import numpy as np
import time
from typing import Optional, Dict, List
from pathlib import Path
import glob
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import uuid
from datetime import datetime

def generate_unique_id() -> str:
    """고유 ID 생성"""
    return str(uuid.uuid4())

# JSONL 파일에서 "text_for_embedding" 필드와 "id" 필드 추출
def load_jsonl_text_and_ids(path, text_field="text_for_embedding", limit=0): ## 실험용 limit 설정 (전체 처리 시 0으로 설정)
    texts = []
    ids = []
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f): # 라인별 json 파싱
            obj = json.loads(line)
            
            # id 필드 추출
            if "id" in obj:
                ids.append(obj["id"])
            else:
                ids.append(f"missing_id_{i}")  # id가 없으면 인덱스 기반 임시 id 생성
            
            # text_for_embedding 필드 추출
            if text_field in obj:
                texts.append(obj[text_field])
            elif text_field in obj.get("metadata", {}):
                texts.append(obj["metadata"][text_field])
            else:
                texts.append("") # 필드가 없으면 빈 문자열로 처리
            
            if limit and len(texts) >= limit:
                break
    
    return texts, ids

# JSONL 파일에서 "text_for_embedding" 필드만 추출 (기존 함수 유지)
def load_jsonl_text(path, field="text_for_embedding", limit=0): ## 실험용 limit 설정 (전체 처리 시 0으로 설정)
    data = []
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f): # 라인별 json 파싱
            obj = json.loads(line)
            if field in obj:
                data.append(obj[field]) # text_for_embedding 필드 추출
            elif field in obj.get("metadata", {}):
                data.append(obj["metadata"][field]) # metadata 필드 추출
            else:
                data.append("") # 필드가 없으면 빈 문자열로 처리
            if limit and len(data) >= limit:
                break
    return data

# ID 매핑 JSON 파일 저장
def save_id_mapping(ids: List[str], output_file: str, input_file: str, model_name: str, pooling_strategy: str):
    """임베딩과 원본 ID의 매핑 정보를 JSON 파일로 저장"""
    mapping_data = {
        "metadata": {
            "input_file": input_file,
            "model_name": model_name,
            "pooling_strategy": pooling_strategy,
            "total_items": len(ids),
            "created_at": datetime.now().isoformat()
        },
        "id_mapping": [
            {"index": i, "id": id_val} 
            for i, id_val in enumerate(ids)
        ]
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"📝 ID 매핑 저장 완료: {output_file}")
    return output_file

# 임베딩 모델 정보 출력
def get_model_info(model_name: str) -> Optional[Dict]:
    info = {"model_name": model_name}

    # Sentence-Transformers로 모델 로딩 시도
    try:
        st_model = SentenceTransformer(model_name)
        dim = st_model.get_sentence_embedding_dimension()

        # 지원하는 pooling 전략 추출 (mean, cls, max)
        pooling = []
        for _, mod in st_model.named_modules():
            cls_name = mod.__class__.__name__.lower()
            if "pooling" in cls_name:
                if getattr(mod, "pooling_mode_mean_tokens", False): pooling.append("mean")
                if getattr(mod, "pooling_mode_cls_token", False):  pooling.append("cls")
                if getattr(mod, "pooling_mode_max_tokens", False):  pooling.append("max")

        # 최적 pooling 전략 선택
        optimal_pooling = "mean"  # 기본값
        if "cls" in pooling:
            optimal_pooling = "cls"  # cls가 있으면 우선 선택 (BERT 계열)
        elif "mean" in pooling:
            optimal_pooling = "mean"  # mean이 있으면 선택
        elif "max" in pooling:
            optimal_pooling = "max"  # max만 있으면 선택

        info.update({
            "backend": "sentence-transformers",
            "embedding_dimension": dim,
            "max_seq_length": st_model.max_seq_length, # 최대 토큰 수
            "supported_pooling": pooling,
            "optimal_pooling": optimal_pooling,
            "padding_side": getattr(st_model.tokenizer, "padding_side", "right"),
        })

        print(f"📊 모델: {model_name}")
        print(f"   백엔드: sentence-transformers")
        print(f"   임베딩 차원: {dim}")
        print(f"   최대 토큰 수: {st_model.max_seq_length}")
        print(f"   지원 Pooling: {', '.join(pooling) if pooling else '없음'}")
        print(f"   최적 Pooling: {optimal_pooling}")
        print(f"   패딩 방향: {info['padding_side']}")
        return info

    except Exception as e:
        print(f"❌ 모델 로딩 오류 {model_name}: {e}")
        return None
    
# 임베딩 생성
def embed_texts(texts, model_name, batch_size=32, device="cpu", pooling_strategy=None):
    print(f"🔄 모델 로딩 중: {model_name}")
    model = SentenceTransformer(model_name, device=device)  # 모델 로드
    
    # pooling 전략 결정
    if pooling_strategy is None:
        # 모델 정보에서 최적 pooling 전략 가져오기
        model_info = get_model_info(model_name)
        if model_info and 'optimal_pooling' in model_info:
            pooling_strategy = model_info['optimal_pooling']
            print(f"🎯 자동 선택된 pooling 전략: {pooling_strategy}")
        else:
            pooling_strategy = "mean"  # 기본값
            print(f"⚠️  기본 pooling 전략 사용: {pooling_strategy}")
    
    print(f"🔧 사용할 pooling 전략: {pooling_strategy}")
    
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"임베딩 생성 중 ({model_name})"):
        batch = texts[i:i + batch_size]
        # 빈 문자열 필터링
        valid_batch = [text for text in batch if text.strip()]
        if valid_batch:
            try:
                # pooling 전략을 명시적으로 지정
                emb = model.encode(valid_batch, convert_to_numpy=True, show_progress_bar=False, 
                                 pooling_strategy=pooling_strategy)
            except Exception as e:
                print(f"⚠️  {pooling_strategy} pooling 실패, 기본값 사용: {e}")
                # pooling 전략 실패 시 기본값으로 재시도
                emb = model.encode(valid_batch, convert_to_numpy=True, show_progress_bar=False)
            
            # 0벡터 패딩 추가
            if len(valid_batch) < len(batch):
                padding = np.zeros((len(batch) - len(valid_batch), emb.shape[1]), dtype=emb.dtype)
                emb = np.vstack([emb, padding])
            embeddings.append(emb)
        else:
            # 모든 텍스트가 빈 경우
            emb = np.zeros((len(batch), model.get_sentence_embedding_dimension()), dtype=np.float32)
            embeddings.append(emb)
    
    return np.vstack(embeddings)  # 최종 임베딩 배열 변환

# 실험 로그 저장 (txt)
def save_experiment_log(results: List[Dict], output_dir: str, experiment_id: str):
    log_data = {
        'experiment_id': experiment_id,
        'timestamp': datetime.now().isoformat(),
        'total_files': len(results),
        'results': results
    }
    
    log_file = Path(output_dir) / f"experiment_log_{experiment_id}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"실험 ID: {log_data['experiment_id']}\n")
        f.write(f"타임스탬프: {log_data['timestamp']}\n")
        f.write(f"전체 파일 수: {log_data['total_files']}\n\n")
        
        for i, result in enumerate(log_data['results'], 1):
            f.write(f"=== 결과 {i} ===\n")
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    print(f"📝 실험 로그 저장 완료: {log_file}")
    return log_file


# 임베딩 실험 실행
def run_embedding_experiment(input_file, output_dir, model_name, batch_size=32, device="cpu", limit=0, pooling_strategy=None):
    start_time = time.time()
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드 (텍스트와 ID 모두 추출)
    print(f"📄 데이터 로딩 중: {input_file}")
    texts, ids = load_jsonl_text_and_ids(input_file, text_field="text_for_embedding", limit=limit)
    print(f"📊 로드된 텍스트 수: {len(texts)}")
    print(f"📊 로드된 ID 수: {len(ids)}")
    
    # 빈 텍스트 필터링
    valid_texts = [text for text in texts if text.strip()]
    print(f"📝 유효한 텍스트: {len(valid_texts)} / {len(texts)}")
    
    print(f"\n🔬 실험 시작: {model_name}")
    print("=" * 60)
    
    try:
        # 모델 정보 출력
        model_info = get_model_info(model_name)
        if model_info is None:
            return None
        
        # pooling 전략 결정
        if pooling_strategy is None:
            pooling_strategy = model_info.get('optimal_pooling', 'mean')
        
        # 임베딩 생성
        embeddings = embed_texts(texts, model_name, batch_size=batch_size, device=device, pooling_strategy=pooling_strategy)
        
        # 결과 저장
        stem = Path(input_file).stem
        model_safe_name = model_name.replace("/", "_").replace("-", "_")
        
        # 임베딩 파일 저장
        embedding_output_file = output_path / f"embeddings_{stem}_{model_safe_name}_{pooling_strategy}.npy"
        np.save(embedding_output_file, embeddings.astype(np.float32))
        print(f"✅ 임베딩 저장 완료: {embedding_output_file} (shape={embeddings.shape})")
        
        # ID 매핑 파일 저장
        id_mapping_file = output_path / f"id_mapping_{stem}_{model_safe_name}_{pooling_strategy}.json"
        save_id_mapping(ids, str(id_mapping_file), str(input_file), model_name, pooling_strategy)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        result = {
            "input_file": str(input_file),
            "model_name": model_name,
            "output_file": str(embedding_output_file),
            "id_mapping_file": str(id_mapping_file),
            "shape": embeddings.shape,
            "model_info": model_info,
            "pooling_strategy": pooling_strategy,
            "valid_texts": len(valid_texts),
            "total_ids": len(ids),
            "processing_time_seconds": processing_time,
            "batch_size": batch_size,
            "device": device,
            "limit": limit
        }
        
        print(f"✅ 실험 완료: {model_name} - {result['shape']}")
        print(f"🔧 Pooling 전략: {pooling_strategy}")
        print(f"📄 ID 매핑: {len(ids)}개")
        print(f"⏱️  처리 시간: {processing_time:.2f}초")
        
        return result
        
    except Exception as e:
        print(f"❌ 실험 오류 {model_name}: {e}")
        return {
            "input_file": str(input_file),
            "model_name": model_name, 
            "error": str(e),
            "processing_time_seconds": time.time() - start_time
        }


def main():
    """메인 함수"""
    print("🚀 TorchDocs 임베딩 생성 시작!")
    print("=" * 60)
    
    # 설정
    input_dir = "TorchDocs/data/preprocessed/"
    output_dir = "TorchDocs/data/embeddings_output"
    model_name = "BAAI/bge-large-en"
    batch_size = 16
    device = "cpu"  ## GPU 사용 시 "cuda"로 변경
    limit = 10     ## 테스트용으로 10개만 처리, 전체 처리 시 0으로 설정
    pooling_strategy = None  # None: 자동 선택 ("mean", "cls", "max" 선택도 가능)
    
    # 모델별 출력 디렉토리 생성
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    model_output_dir = Path(output_dir) / model_safe_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 모델별 출력 디렉토리: {model_output_dir}")
    
    # 실험 ID 생성
    experiment_id = generate_unique_id()
    print(f"🔬 실험 ID: {experiment_id}")
    
    # pooling 전략 정보 출력
    if pooling_strategy:
        print(f"🔧 사용자 지정 Pooling: {pooling_strategy}")
    else:
        print(f"🎯 Pooling: 자동 선택 (모델별 최적 전략)")
    
    # 폴더 내의 모든 JSONL 파일 검색
    input_files = glob.glob(str(Path(input_dir) / "*.jsonl"))
    print(f"📁 발견된 JSONL 파일 수: {len(input_files)}")
    for file in input_files:
        print(f"  - {file}")
    
    if not input_files:
        print("❌ 처리할 JSONL 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n🚀 모든 JSONL 파일에 대한 임베딩 실험 시작!")
    print(f"📊 모델: {model_name}")
    print(f"⚙️  배치 크기: {batch_size}")
    print(f"💻 디바이스: {device}")
    print(f"📏 제한: {'전체' if limit == 0 else f'{limit}개'}")
    
    # 각 파일에 대해 임베딩 실험 실행
    all_results = []
    total_start_time = time.time()
    
    for input_file in input_files:
        print(f"\n📄 처리 중: {input_file}")
        file_result = run_embedding_experiment(
            input_file=input_file,
            output_dir=str(model_output_dir),
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            limit=limit,
            pooling_strategy=pooling_strategy
        )
        if file_result:
            all_results.append(file_result)
    
    total_time = time.time() - total_start_time
    
    print(f"\n✅ 모든 파일 처리 완료!")
    print(f"📊 총 결과 수: {len(all_results)}")
    print(f"⏱️  총 소요 시간: {total_time:.2f}초")
    
    # 실험 로그 저장
    if all_results:
        log_file = save_experiment_log(all_results, str(model_output_dir), experiment_id)
        print(f"📝 실험 요약:")
        print(f"   로그 파일: {log_file}")
        print(f"   성공한 실험: {len([r for r in all_results if 'error' not in r])}")
        print(f"   실패한 실험: {len([r for r in all_results if 'error' in r])}")
        
        # pooling 전략별 결과 요약
        pooling_results = {}
        for result in all_results:
            if 'error' not in result:
                strategy = result.get('pooling_strategy', 'unknown')
                if strategy not in pooling_results:
                    pooling_results[strategy] = 0
                pooling_results[strategy] += 1
        
        if pooling_results:
            print(f"🔧 Pooling 전략별 결과:")
            for strategy, count in pooling_results.items():
                print(f"   {strategy}: {count}개 파일")
        
        # ID 매핑 파일 생성 요약
        successful_results = [r for r in all_results if 'error' not in r]
        if successful_results:
            print(f"📝 생성된 파일:")
            for result in successful_results:
                print(f"   임베딩: {Path(result['output_file']).name}")
                print(f"   ID 매핑: {Path(result['id_mapping_file']).name}")
                print(f"   ID 개수: {result['total_ids']}개")
    
    print("\n🎉 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()