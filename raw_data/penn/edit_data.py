import pickle
from pathlib import Path

data_folder = Path('raw_data/penn/data_kyle_clean')
files = sorted(data_folder.glob('*.pkl'))

hardcoded_poses = {
    0: (5.457786630525433, 4.300992717999654, 3.8020802208157227),
    1: (4.830377610214883, 4.590675708835122, -3.8111006617871492),
    2: (4.6783017558682936, 4.962070082403231, -4.2953699357435955),
    3: (4.719521826973929, 5.025721949187286, 1.730678314959284),
    4: (4.978490975967743, 4.909736328470612, 0.49034190364374797),
    5: (5.055225445891463, 4.85195270157447, 0.14724175335798267),
    6: (5.549872105320095, 5.282100316597524, 0.685983249848645),
    7: (5.9461142354979595, 5.703166778295573, 0.7754690971716794),
    8: (6.01632514568523, 5.702254030073702, 0.5947371790789254),
    9: (6.016201561082811, 5.664731888828451, 0.4091565218840698),
    10: (5.733720520417075, 5.784741282078776, -4.268490395600127)
}


def load_pickle(path: Path):
    path = Path(path)
    assert path.exists(), f'Path {path} does not exist.'
    with open(path, 'rb') as f:
        return pickle.load(f, encoding="latin1")


def save_pickle(path: Path, data):
    path = Path(path)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


for idx, file in enumerate(files):
    if idx not in hardcoded_poses:
        continue
    print(f'Processing file {file}')
    scan, pose = load_pickle(file)
    pose.x, pose.y, pose.theta = hardcoded_poses[idx]
    save_pickle(file, (scan, pose))