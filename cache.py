import pickle
from os import path, makedirs
import hashlib

def is_filesame(files: list) -> bool:
    for file, checksum in files:
        with open(file, 'rb') as f:
            data = f.read()
            if checksum != hashlib.md5(data).hexdigest():
                return False
    return True

def load_cache(key):
    try:
        filepath = path.join('cache', f'{key}.pkl')
        if path.isfile(filepath):
            with open(filepath, 'rb') as f:
                cache_obj = pickle.load(f)
                if type(cache_obj) is dict:
                    if 'watch' in cache_obj and 'obj' in cache_obj and type(cache_obj['watch']) is list and is_filesame(cache_obj['watch']):
                        return cache_obj['obj']
    except:
        pass
    return None

def store_cache(key, watch, obj):
    if not path.exists('cache'):
        makedirs('cache')
    filepath = path.join('cache', f'{key}.pkl')

    watch_cache = []
    for file in watch:
        with open(file, 'rb') as f:
            data = f.read()
            watch_cache.append((file, hashlib.md5(data).hexdigest()))
    
    cache_obj = dict()
    cache_obj['watch'] = watch_cache
    cache_obj['obj'] = obj

    with open(filepath, 'wb') as f:
        pickle.dump(cache_obj, f)
    
    return obj


