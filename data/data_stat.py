'''
    helper functions to read the TSM_feature lmdb
    run this with a command line argument describing the path to the lmdb
    e.g. python read_lmdb.py TSM_features/C10095_rgb
'''
import os
import sys
import lmdb
import numpy as np
import pickle as pkl

# path to the lmdb file you want to read as a command line argument
lmdb_path = sys.argv[1]

# iterate over the entire lmdb and output all files
def extract_all_features(env, view):
    '''
        input:
            env: lmdb environment loaded (see main function)
        output: a dictionary with key as the path_to_frame and value as the TSM feature (2048-D np-array)
                the lmdb key format is '{sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg'
                e.g. nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724/C10095_rgb/C10095_rgb_0000000001.jpg
    '''
    # ALL THE FRAME NUMBERS ARE AT 30FPS !!!

    results = {}  # set()
    print('Iterating over the entire lmdb. This may take some time...')
    i= 0
    with env.begin() as e:
        cursor = e.cursor()
        for file, data in cursor:
            # if i>100000: break
            # i+=1
            vid = file.decode("utf-8")
            data = np.frombuffer(data, dtype=np.float32)
            if data.shape[0] == 1536:
                vid_name, img_name = vid.split('_')[0], vid.split('_')[1]
                view_name = 'exo'
                assert view_name == view
                if vid_name not in results:
                    results[vid_name] = {}
                if view_name not in results[vid_name]:
                    results[vid_name][view_name] = []
                frame_id = img_name.replace(view_name, '').replace('_', '').split('.')[0]
                results[vid_name][view_name].append(int(frame_id))
            else:
                print(vid, data.shape)

    final_results = {}
    minimum_f, maximum_f = [], []
    for vid_name in results.keys():
        if vid_name not in final_results:
            final_results[vid_name] = {}
        for view_name in results[vid_name]:
            if view_name in final_results[vid_name]:
                print('files in views should not be handled yet')
                exit(1)
            max_id, min_id = max(results[vid_name][view_name]), min(results[vid_name][view_name])
            maximum_f.append(max_id)
            minimum_f.append(min_id)
            assert (max_id - min_id + 1) == len(results[vid_name][view_name])
            final_results[vid_name][view_name] = [min_id, max_id + 1]


    print(f'Features [view {view} ]: {len(final_results)} tasks loaded.')
    print('videos start at:', np.unique(np.array(minimum_f)))
    print('max length video with frames end at', np.max(np.array(maximum_f)))

    return final_results

def merge_dict(A, B):
    # B=> A
    if len(A) == 0:
        return B
    if len(B) == 0:
        return A
    for vid_name in B.keys():
        assert len(B[vid_name]) == 1
        view = list(B[vid_name].keys())[0]
        if vid_name in A:
            assert view not in A[vid_name]
            A[vid_name][view] = B[vid_name][view]
        else:
            print(vid_name, "not in the current dict")
            A[vid_name] = B[vid_name]
    return A



def save_statistic_input(views):
    total = {}
    for view in views:
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        final_results = extract_all_features(env, view)
        total = merge_dict(total, final_results)

    with open('statistic_input_exo.pkl', 'wb') as f:
        pkl.dump(total, f)


# main function
if __name__ == '__main__':
    #views = ['ego']
    views = ['exo']

    save_statistic_input(views)
