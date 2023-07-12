from fastapi import APIRouter, File, UploadFile, BackgroundTasks
import numpy as np

router = APIRouter()


@router.post("/suggest/")
def get_recommendation(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    names, scores_ = [], []
    ucb_scores = []
    n_total = 0
    fpath = 'scores.csv'
    with open(fpath, 'r') as f:
        line = f.readline()
        while line:
            parts = line.replace('\n', '').split(',')

            names.append(parts[0])
            scores_.append(np.array([int(a) for a in parts[1:]]))
            n_total += len(parts[1:])

            line = f.readline()

    for scores in scores_:
        score = np.mean(scores) + np.sqrt(2 * np.log(n_total) / len(scores))
        ucb_scores.append(score)

    best = names[np.array(ucb_scores).argmax()]

    background_tasks.add_task(file.file.close)

    return best
