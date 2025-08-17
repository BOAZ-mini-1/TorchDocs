import json
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import glob
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime


def load_jsonl_text_and_ids(path, text_field="text_for_embedding"):
    """JSONL 파일에서 "text_for_embedding" 필드와 "id" 필드 추출"""
    texts = []
    ids = []
    
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            
            # id 필드 추출
            if "id" in obj:
                ids.append(obj["id"])
            else:
                ids.append(f"missing_id_{i}")
            
            # text_for_embedding 필드 추출
            if text_field in obj:
                texts.append(obj[text_field])
            elif text_field in obj.get("metadata", {}):
                texts.append(obj["metadata"][text_field])
            else:
                texts.append("")
    
    return texts, ids


def get_model_max_length(model_name: str) -> int:
    """SentenceTransformer 모델의 max_seq_length 가져오기"""
    try:
        st_model = SentenceTransformer(model_name)
        max_length = st_model.max_seq_length
        print(f"📏 모델 최대 시퀀스 길이: {max_length}")
        return max_length
    except Exception as e:
        print(f"⚠️ 모델 로드 실패, 기본값 512 사용: {e}")
        return 512


def check_token_length(texts: List[str], ids: List[str], model_name: str) -> Dict:
    """모든 텍스트의 토큰 수 확인 및 모델 제한 초과 여부 분석"""
    print(f"🔍 모든 텍스트의 토큰 수 분석 중: {model_name}")
    
    try:
        # SentenceTransformer 모델에서 최대 길이 가져오기
        max_length = get_model_max_length(model_name)
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        token_counts = []
        problematic_items = []
        
        for i, (text, item_id) in enumerate(tqdm(zip(texts, ids), desc="토큰 수 계산 중", total=len(texts))):
            if text.strip():
                tokens = tokenizer.encode(text, add_special_tokens=True)
                token_count = len(tokens)
                token_counts.append(token_count)
                
                if token_count > max_length:
                    problematic_items.append({
                        'index': i,
                        'id': item_id,
                        'text_preview': text[:200] + "..." if len(text) > 200 else text,
                        'token_count': token_count,
                        'max_length': max_length,
                        'excess_tokens': token_count - max_length
                    })
            else:
                token_counts.append(0)
        
        analysis = {
            'total_items': len(texts),
            'valid_items': len([t for t in texts if t.strip()]),
            'max_tokens': max(token_counts) if token_counts else 0,
            'avg_tokens': np.mean(token_counts) if token_counts else 0,
            'median_tokens': np.median(token_counts) if token_counts else 0,
            'items_exceeding_limit': len(problematic_items),
            'additional_chunking_needed': len(problematic_items) > 0,
            'problematic_items': problematic_items,
            'model_max_length': max_length
        }
        
        print(f"📊 토큰 분석 결과:")
        print(f"   총 아이템 수: {analysis['total_items']}")
        print(f"   유효한 아이템: {analysis['valid_items']}")
        print(f"   최대 토큰 수: {analysis['max_tokens']}")
        print(f"   평균 토큰 수: {analysis['avg_tokens']:.1f}")
        print(f"   중위값 토큰 수: {analysis['median_tokens']:.1f}")
        print(f"   모델 최대 길이: {analysis['model_max_length']}")
        print(f"   제한 초과 아이템: {analysis['items_exceeding_limit']}")
        print(f"   추가 청킹 필요: {'예' if analysis['additional_chunking_needed'] else '아니오'}")
        
        if analysis['additional_chunking_needed']:
            print(f"⚠️  모델 제한을 초과하는 아이템들:")
            for item in problematic_items:
                print(f"   ID: {item['id']} | 토큰: {item['token_count']} (초과: {item['excess_tokens']})")
        
        return analysis
        
    except Exception as e:
        print(f"❌ 토큰 분석 오류: {e}")
        return None


def save_token_analysis_log(analysis_results: List[Dict], output_dir: str, timestamp: str):
    """토큰 분석 결과를 JSON 파일로 저장"""
    log_data = {
        'timestamp': timestamp,
        'total_files': len(analysis_results),
        'analysis_results': analysis_results
    }
    
    log_file = Path(output_dir) / f"token_analysis_log_{timestamp.replace(':', '-').replace('.', '-')}.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"📝 토큰 분석 로그 저장 완료: {log_file}")
    return log_file


def analyze_tokens_for_file(input_file: str, model_name: str) -> Dict:
    """단일 파일에 대한 모든 텍스트 토큰 분석"""
    print(f"📄 모든 텍스트 토큰 분석 중: {input_file}")
    
    try:
        # 데이터와 ID 로드
        texts, ids = load_jsonl_text_and_ids(input_file, text_field="text_for_embedding")
        print(f"📊 로드된 아이템 수: {len(texts)}")
        
        # 빈 텍스트 필터링
        valid_texts = [text for text in texts if text.strip()]
        print(f"📝 유효한 아이템: {len(valid_texts)} / {len(texts)}")
        
        # 토큰 분석
        token_analysis = check_token_length(texts, ids, model_name)
        
        if token_analysis:
            result = {
                "input_file": str(input_file),
                "model_name": model_name,
                "token_analysis": token_analysis,
                "valid_items": len(valid_texts)
            }
            return result
        else:
            return {
                "input_file": str(input_file),
                "model_name": model_name,
                "error": "텍스트 토큰 분석 실패"
            }
            
    except Exception as e:
        print(f"❌ 파일 분석 오류 {input_file}: {e}")
        return {
            "input_file": str(input_file),
            "model_name": model_name,
            "error": str(e)
        }


def main():
    """메인 함수 - 모든 텍스트의 토큰 분석 실행"""
    print("🔍 TorchDocs 전체 텍스트 토큰 분석 시작!")
    print("=" * 60)
    print("📋 목적: 모든 text_for_embedding의 토큰 수 확인")
    print("📋 대상: preprocessed 폴더 내 모든 JSONL 파일의 모든 텍스트")
    print("📋 확인사항: 텍스트가 모델의 최대 토큰 수를 초과하는지 여부")
    print("📋 결과: 문제 없으면 make_embeddings.py 실행, 문제 있으면 추가 청킹 필요")
    print("=" * 60)
    
    # 설정
    input_dir = "TorchDocs/data/preprocessed/"
    output_dir = "TorchDocs/token_analysis_output"
    model_name = "BAAI/bge-large-en"
    
    # 분석 타임스탬프 생성
    timestamp = datetime.now().isoformat()
    print(f"🔬 분석 시작 시간: {timestamp}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 폴더 내의 모든 JSONL 파일 검색
    input_files = glob.glob(str(Path(input_dir) / "*.jsonl"))
    print(f"📁 발견된 JSONL 파일 수: {len(input_files)}")
    for file in input_files:
        print(f"  - {file}")
    
    if not input_files:
        print("❌ 분석할 JSONL 파일을 찾을 수 없습니다.")
        return
    
    print(f"\n🔍 모든 JSONL 파일의 전체 텍스트 토큰 분석 시작!")
    print(f"📊 모델: {model_name}")
    
    # 각 파일에 대해 토큰 분석 실행
    all_results = []
    total_problematic_items = []
    
    for input_file in input_files:
        print(f"\n📄 처리 중: {input_file}")
        file_result = analyze_tokens_for_file(
            input_file=input_file,
            model_name=model_name
        )
        if file_result:
            all_results.append(file_result)
            
            # 문제가 있는 아이템들 수집
            if 'token_analysis' in file_result and file_result['token_analysis']['problematic_items']:
                for item in file_result['token_analysis']['problematic_items']:
                    item['source_file'] = input_file
                    total_problematic_items.append(item)
    
    print(f"\n✅ 모든 파일의 전체 텍스트 토큰 분석 완료!")
    print(f"📊 총 결과 수: {len(all_results)}")
    
    # 분석 로그 저장
    if all_results:
        log_file = save_token_analysis_log(all_results, output_dir, timestamp)
        print(f"📝 분석 요약:")
        print(f"   로그 파일: {log_file}")
        print(f"   성공한 분석: {len([r for r in all_results if 'error' not in r])}")
        print(f"   실패한 분석: {len([r for r in all_results if 'error' in r])}")
        
        # 추가 청킹 필요성 요약
        additional_chunking_needed_count = sum([
            1 for r in all_results 
            if 'token_analysis' in r and r['token_analysis']['additional_chunking_needed']
        ])
        print(f"   추가 청킹이 필요한 파일: {additional_chunking_needed_count}개")
        print(f"   전체 문제 아이템 수: {len(total_problematic_items)}개")
        
        # 문제가 있는 아이템들의 상세 정보
        if total_problematic_items:
            print(f"\n⚠️  토큰 제한 초과 아이템 상세 정보:")
            for i, item in enumerate(total_problematic_items, 1):
                print(f"   {i}. 파일: {item['source_file']}")
                print(f"      ID: {item['id']}")
                print(f"      토큰 수: {item['token_count']} (초과: {item['excess_tokens']})")
                print(f"      미리보기: {item['text_preview']}")
                print()
        
        # 최종 권장사항
        print(f"\n🎯 최종 권장사항:")
        if len(total_problematic_items) == 0:
            print(f"   ✅ 모든 텍스트가 모델 제한 내에 있습니다.")
            print(f"   ✅ 현재 상태로 make_embeddings.py를 실행할 수 있습니다.")
        else:
            print(f"   ⚠️  총 {len(total_problematic_items)}개 아이템에서 모델 제한 초과가 발견되었습니다.")
            print(f"   ⚠️  추가 청킹 기능 구현이 필요합니다.")
            print(f"   ⚠️  make_embeddings.py 실행 전에 전처리 단계에서 더 세밀한 청킹이 필요합니다.")
            print(f"   📋 문제 아이템들의 상세 정보는 위에서 확인하세요.")
    
    print("\n🎉 전체 텍스트 토큰 분석이 완료되었습니다!")


if __name__ == "__main__":
    main()
