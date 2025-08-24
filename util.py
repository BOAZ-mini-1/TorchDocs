# import zipfile
# from pathlib import Path
#
#
# def find_path(name, search_directory='../', path_type='any'):
#     search_path = Path(search_directory)
#
#     for path in search_path.rglob(name):
#         if path_type == 'dir' and path.is_dir():
#             return str(path.resolve())
#         elif path_type == 'file' and path.is_file():
#             return str(path.resolve())
#         elif path_type == 'any':
#             return str(path.resolve())
#
#     return None
#
#
# def find_all_paths(pattern, search_directory):
#     search_path = Path(find_path(search_directory))
#
#     return [str(p.resolve()) for p in search_path.glob(pattern)]
#
#
# def unzip_if_needed(zip_file_path: str):
#
#     zip_path = Path(find_path(zip_file_path))
#
#     if not zip_path.is_file():
#         print(f"오류: '{zip_file_path}' 파일을 찾을 수 없습니다.")
#         return
#
#     try:
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             namelist = zip_ref.namelist()
#
#         if not namelist:
#             print(f"'{zip_path.name}' 파일은 비어있어 작업을 건너뜁니다.")
#             return
#
#         first_item_path = zip_path.parent / namelist[0]
#
#         if first_item_path.exists():
#             print(f"'{first_item_path.name}'이(가) 이미 존재하므로 압축을 건너뜁니다.")
#             return
#
#         extraction_path = zip_path.parent
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(extraction_path)
#
#         print(f"'{zip_path.name}' 파일의 내용물을 '{extraction_path}' 폴더에 성공적으로 해제했습니다.")
#
#     except zipfile.BadZipFile:
#         print(f"오류: '{zip_path.name}'은(는) 유효한 ZIP 파일이 아닙니다.")
#     except Exception as e:
#         print(f"오류 발생: {e}")